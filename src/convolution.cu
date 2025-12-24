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

void printImageData(unsigned char **input, int width, int height);
double calc_seconds(struct timeval *start_time, struct timeval *end_time);
void rec_exectime_vs_blockdim(char *filename, int dim, int kernelSize, double exectime);

__global__ void convolution_kernel(unsigned char *input_img, int *kernel,
                                   unsigned char *output_img, int width,
                                   int height, int kernelSize) {

  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= height || col >= width) {
    return;
  }

  // boundaries directly set to 0
  if (row == 0 || row == (height - 1) || col == 0 || col >= (width - 3)) {
    output_img[row * width + col] = 0;
    return;
  }

  int sum = 0;

  for (int kcol = 0; kcol < kernelSize; kcol++) {
    for (int krow = 0; krow < kernelSize; krow++) {
      int PixelVal = input_img[(row + krow - 1) * width + (col + kcol - 1)];
      sum += PixelVal * kernel[krow * kernelSize + kcol];
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

  const char *outputFile = "examples/outputs/global/baldursgate_convolution.png";

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
  size_t kernel_size = KERNEL_DIM * KERNEL_DIM * sizeof(int);

  // host output allocation
  unsigned char *h_output = (unsigned char *)malloc(img_size);

  // device input/output memory allocations
  unsigned char *d_input, *d_output;
  int *d_kernel;

  cudaMalloc(&d_input, img_size);
  cudaMalloc(&d_output, img_size);
  cudaMalloc(&d_kernel, kernel_size);

  cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

  dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  int num_blocks = gridSize.x * gridSize.y;
  int num_threads = blockSize.x * blockSize.y;

  printf("Number of blocks: %d\n", num_blocks);
  printf("Number of threads: %d\n", num_threads);

  struct timeval start_time, end_time;

  gettimeofday(&start_time, NULL);
  convolution_kernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                              width, height, KERNEL_DIM);
  cudaDeviceSynchronize();
  gettimeofday(&end_time, NULL);

  double exec_time = calc_seconds(&start_time, &end_time);
  printf("Process took %lf seconds\n", exec_time);
  char *filename = (char*) "global.dat";
  rec_exectime_vs_blockdim(filename, BLOCK_DIM, KERNEL_DIM, exec_time);

  cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

  stbi_write_png(outputFile, width, height, 1, h_output, width);
  printf("Output image written to the file %s\n", outputFile);
  stbi_image_free(h_input);

  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_kernel);

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
