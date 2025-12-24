#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../includes/blockdim.h"
#include "../includes/stb_image.h"
#include "../includes/stb_image_write.h"
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define KERNEL_DIM 3
#define HALO (KERNEL_DIM / 2)
#define TILE_DIM (BLOCK_DIM + 2 * HALO)

void printImageData(unsigned char **input, int width, int height);
double calc_seconds(struct timeval *start_time, struct timeval *end_time);
void rec_exectime_vs_blockdim(char *filename, int dim, int kernelSize, double exectime);


__constant__ int c_kernel[KERNEL_DIM * KERNEL_DIM];

__global__ void convolution_kernel(unsigned char *input_img,
                                   unsigned char *output_img, int width,
                                   int height) {

  __shared__ unsigned char s_tile[TILE_DIM][TILE_DIM];

  // y for the row since it is height and x for column
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Load data to shared
  if (row >= height || col >= width) {
    s_tile[threadIdx.y + HALO][threadIdx.x + HALO] = 0;
  }
  else {
    s_tile[threadIdx.y + HALO][threadIdx.x + HALO] = input_img[row * width + col];
  }
 
  // load halo pixels
  // left halo
  if (threadIdx.x < HALO) {
    int c = col - HALO;
    if (c < width && c >= 0) {
      s_tile[threadIdx.y + HALO][threadIdx.x] = input_img[row * width + c];
    }
    else {
      s_tile[threadIdx.y + HALO][threadIdx.x] = 0;
    }
  }

  // top halo
  if (threadIdx.y < HALO) {
    int r = row - HALO;
    if (r >= 0) {
      s_tile[threadIdx.y][threadIdx.x + HALO] = input_img[r * width + col];
    }
    else {
      s_tile[threadIdx.y][threadIdx.x + HALO] = 0;
    }
  }
  
  // right halo
  if (threadIdx.x >= BLOCK_DIM - HALO) { // tx == 15
     int c = col + HALO;
     // Map thread 15 to s_tile index 17 (15 + 2*R)
     if (c < width && row < height) {
         s_tile[threadIdx.y + HALO][threadIdx.x + 2 * HALO] = input_img[row * width + c];
     }
     else {
         s_tile[threadIdx.y + HALO][threadIdx.x + 2 * HALO] = 0;
     }
  }

  // bottom halo
  if (threadIdx.y >= BLOCK_DIM - HALO) {
    int r = row + HALO;
    if (r < height && r > 0) {
      s_tile[threadIdx.y + 2 * HALO][threadIdx.x + HALO] = input_img[r * width + col];
    }
    else {
      s_tile[threadIdx.y + 2 * HALO][threadIdx.x + HALO] = 0;
    }
  }
  
  // synchronize threads so that they won't write each other's place
  // this may slow down the process
  __syncthreads();

  // boundaries directly set to 0
  if (row == 0 || row == (height - 1) || col == 0 || col == (width - 1)) {
    output_img[row * width + col] = 0;
    return;
  }

  if (row >= height || col >= width) {
    return;
  }
  
  int sum = 0;
  
  // kcol geos from 0 to 2
  for (int kcol = 0; kcol < KERNEL_DIM; kcol++) {
    for (int krow = 0; krow < KERNEL_DIM; krow++) {
      // threadIdx.y & .x go from 0 to 15
      // srow & scol go from -1 to 16
      int srow = threadIdx.y + krow;
      int scol = threadIdx.x + kcol;

      if (!(srow < 0 || (srow >= TILE_DIM)|| scol < 0 || (scol >= TILE_DIM))) {
        int pixel = (int) s_tile[srow][scol];
        int weight = c_kernel[krow * KERNEL_DIM + kcol];
        sum += pixel * weight;
      }
    }
  }


  if (sum < 0) {
    sum = 0;
  }
  if (sum > 255) {
    sum = 255;
  }

  output_img[row * width + col] = (unsigned char)sum;
}

int main() {
  const char *inputFile = "examples/inputs/baldursgate.png";

  const char *outputFile = "examples/outputs/shared/baldursgate_convolution.png";

  int width, height, channels;

  // laplacian kernel
  int h_kernel[KERNEL_DIM * KERNEL_DIM] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

  unsigned char *h_input = stbi_load(inputFile, &width, &height, &channels, 1);
  if (h_input == NULL) {
    perror("ERROR::LOADING_IMAGE\n");
    return -1;
  }
  printf("Image loaded successfully! width: %d, height: %d\n", width, height);

  size_t img_size = width * height * sizeof(unsigned char);

  // host output allocation
  unsigned char *h_output = (unsigned char *)malloc(img_size);

  // device input/output memory allocations
  unsigned char *d_input, *d_output;

  cudaMalloc(&d_input, img_size);
  cudaMalloc(&d_output, img_size);

  cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

  // Use constant memory for kernel storage
  
  cudaError_t err = cudaMemcpyToSymbol(c_kernel, h_kernel, sizeof(h_kernel));
  if (err != cudaSuccess) {
    printf("Error copying to constant memory: %s\n", cudaGetErrorString(err));
    return -1;
  }

  dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  int num_blocks = gridSize.x * gridSize.y;
  int num_threads = blockSize.x * blockSize.y;

  printf("Number of blocks: %d\n", num_blocks);
  printf("Number of threads: %d\n", num_threads);

  struct timeval start_time, end_time;

  gettimeofday(&start_time, NULL);
  convolution_kernel<<<gridSize, blockSize>>>(d_input, d_output,
                                              width, height);
  err = cudaDeviceSynchronize();
  
  gettimeofday(&end_time, NULL);

  if (err != cudaSuccess) {
    printf("ERROR: %s\n", cudaGetErrorString(err));
    return -1;
  }



  double exec_time = calc_seconds(&start_time, &end_time);
  printf("Process took %lf seconds\n", exec_time);
  char *filename = (char *)"shared.dat";

  rec_exectime_vs_blockdim(filename, BLOCK_DIM, KERNEL_DIM, exec_time);

  cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

  stbi_write_png(outputFile, width, height, 1, h_output, width);
  printf("Output image written to the file %s\n", outputFile);
  stbi_image_free(h_input);

  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}

void printImageData(unsigned char **input, int width, int height) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      printf("%d, ", input[i][j]);
    }
    printf("\n");
  }
}

double calc_seconds(struct timeval *start_time, struct timeval *end_time) {
  long unit_transform = 1000000;
  long sec_dif = end_time->tv_sec - start_time->tv_sec;
  long msec_dif = end_time->tv_usec - start_time->tv_usec;
  long total_dif = sec_dif * unit_transform + msec_dif;
  if (total_dif < 0) {
    total_dif *= -1;
  }

  double total_dif_sec = (double)total_dif / unit_transform;

  return total_dif_sec;
}

void rec_exectime_vs_blockdim(char *filename, int dim, int kernelSize, double exectime) {
  char full_filename[512] = "./time_measurements/";
  strcat(full_filename, filename);
  FILE *fp = fopen(full_filename, "a");

  fprintf(fp, "%d\t%d\t%lf\n", dim, kernelSize, exectime);

  fclose(fp);
  return;
}
