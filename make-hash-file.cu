#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"

#define PASSWORD_LENGTH 6

#define FIFTH_POWER (26 * 26 * 26 * 26 * 26)
#define FOURTH_POWER (26 * 26 * 26 * 26)
#define THIRD_POWER (26 * 26 * 26)
#define SECOND_POWER (26 * 26)
#define FIRST_POWER 26

__device__ int POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};
typedef struct output{
  char password[PASSWORD_LENGTH+1];
  char hash[17];
}output_t;

__global__ void hash_it_up (output_t * output, int max){
  
  int tempNum;
  for(int count = 0; count < max; count++){
    char word[] = "aaaaaa";
    tempNum = count;

    //Generate the permutation for the given thread/core
    for(int i = PASSWORD_LENGTH - 1; i >= 0; i--){
      int temp = (int) (tempNum/POWER_ARR[i]);
      word[PASSWORD_LENGTH -1 - i] += temp;
      tempNum = tempNum % POWER_ARR[i];
    }
    word[PASSWORD_LENGTH] = '\0';;
    memcpy(output[count].password, word, sizeof(char)*PASSWORD_LENGTH+1);
    uint candidate_hash[17];
    
    md5((uint*)word, candidate_hash);
    candidate_hash[16] = '\0';
               
    
    memcpy(output[count].hash, candidate_hash, sizeof(char)*17);
    
  }

  
  
}

int main(int argc,char* args[]){
  FILE * file;
  int max = 10;
  output_t * gpu_input;
  output_t * output = (output_t *) malloc(sizeof(output_t)*max);
 
  if(cudaMalloc(&gpu_input, sizeof(output_t)*max) != cudaSuccess){
    perror("Cuda Malloc Failed\n");
  }
  hash_it_up<<<1,1>>>(gpu_input, max);

  if(cudaDeviceSynchronize() != cudaSuccess){
    perror("Cuda Sync Failed\n");
  }

  if(cudaMemcpy(output, gpu_input, sizeof(output_t)*max, cudaMemcpyDeviceToHost) != cudaSuccess){
    perror("Cuda Memcpy Failed Here\n");
    exit(2);
  }
  
  file = fopen("outputFile.txt", "w");
  for(int i = 0; i < max; i++){
    fprintf(file, "%s %s\n",output[i].password, output[i].hash);
  }
  fclose(file);
  
  
  return 0;
}
