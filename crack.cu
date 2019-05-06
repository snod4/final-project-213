#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"

#define NUM_THREADS 128
#define MAX_USERNAME_LENGTH 64

#define PASSWORD_LENGTH 6
#define SIXTH_POWER  (26 * 26 * 26 * 26 * 26* 26)
#define FIFTH_POWER (26 * 26 * 26 * 26 * 26)
#define FOURTH_POWER (26 * 26 * 26 * 26)
#define THIRD_POWER (26 * 26 * 26)
#define SECOND_POWER (26 * 26)
#define FIRST_POWER 26

#define HASH_LENGTH 32

__device__ size_t POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};


typedef struct hashInfo{
  char  password[7];
  uint  hash[4];
  int found = 0;
}hashInfo_t;


//Crack is a function that runs on the GPU and brute forces given hashes.
__global__  void crack(hashInfo_t * hashData, int length){
  //get string permuation
  size_t tempNum =((size_t) blockIdx.x) * ((size_t) NUM_THREADS) +((size_t) threadIdx.x);
  char word[] = "aaaaaa";

  //Generate the permutation for the given thread/core
  for(int i = PASSWORD_LENGTH - 1; i >= 0; i--){
    size_t temp =  tempNum/(POWER_ARR[i]);
    word[5 - i] += temp;
    tempNum = tempNum % POWER_ARR[i];
  }   

  //Calculate the hash with the function md5.
  uint candidate_hash[4];
  md5((uint*)word, candidate_hash);

  //Compare the provided hash to the calculated candidate hash.
  for(int j = 0; j < length; j++){
    if(!(hashData[j].found)
       && candidate_hash[0] == hashData[j].hash[0]
       && candidate_hash[1] == hashData[j].hash[1]
       && candidate_hash[2] == hashData[j].hash[2]
       && candidate_hash[3] == hashData[j].hash[3]){
      memcpy(hashData[j].password, word, PASSWORD_LENGTH+1);
      hashData[j].found = 1;
      break;
    }
  }
  
}

//add hash to hash table
/* void addToTable(hashInfo * table, char * hash){ */
/*   hashInfo_t * temp = (hashInfo_t *) malloc(sizeof(hashInfo_t)); */
/*   strncpy(temp->hash, hash, PASSWORD_LENGTH); */
/*   temp->next = NULL; */
/*   temp->password = NULL; */
  
/*   if(table[hash[0] - 48 ]== NULL){ */
/*     table[hash[0] - 48] = temp; */
/*   } */
/*   else{ */
/*     temp->next =  table[hash[0] - 48]; */
/*     table[hash[0] - 48] = temp; */
/*   } */
/* } */

int main(int argv, char* args[]){
  /*  hashInfo_t * hashTable[74];
      int count = 0;
      //get hashes in here -- add them -- count them //
  
      hashInfo_t * gpu_hashTable;
 
      //ISSUE IN COPYING A LINKED LIST TO THE GPU
      */
  FILE * file = fopen("outputFile.txt", "r");/////////////CHANGE THIS TO ARGV EVENTUALLY/////////////
  int NUM_INPUT = 0;
  fscanf(file, "%d", &NUM_INPUT);
  printf("Number of Input: %d\n", NUM_INPUT);
  
  int number_of_blocks = (308915776+NUM_THREADS)/NUM_THREADS;
  hashInfo_t arr[NUM_INPUT];
  
  
  uint hash[4];
  int count = 0;
  char trash_can[7];

  //Grab the input hashes from a file specified by the user in argv[1].
  while(fscanf(file, "%s", trash_can) != EOF){
    for(int i = 0; i < 4; i++){
      fscanf(file, "%u", &hash[i]);
    }
    memcpy(arr[count].hash, hash, sizeof(uint)*4);
    count++;
  }

  //Create the data structure to pass to the GPU
  hashInfo_t * gpu_arr;
  if(cudaMalloc(&gpu_arr, sizeof(hashInfo_t) * NUM_INPUT) != cudaSuccess){
    perror("Cuda Malloc Failed\n");
    exit(2);
  }

  //Copy over our provided hashes in arr to the GPU_arr for analysis.
  if(cudaMemcpy(gpu_arr, arr, sizeof(hashInfo_t) * NUM_INPUT, cudaMemcpyHostToDevice) != cudaSuccess){
    perror("Cuda CPU to GPU memcpy Failed\n");
    exit(2);
  }

  //Crack the provided hashes on the GPU
  crack<<<number_of_blocks, NUM_THREADS>>>(gpu_arr,NUM_INPUT);

  //Ensure all CUDA threads have terminated
  if(cudaDeviceSynchronize() != cudaSuccess){
    perror("CUDA Thread Synchronization Error\n");
    exit(2);
  }

  //Copy back the cracked passwords from the GPU.
  if(cudaMemcpy(arr, gpu_arr, sizeof(hashInfo_t) * NUM_INPUT, cudaMemcpyDeviceToHost) != cudaSuccess){
    perror("Cuda GPU to CPU memcpy Failed\n");
    exit(2);
  }

  //Print the cracked passwords. Eventually we should delete this and automate
  //password cracking sucess when we scale up the amount of passwords to crack
  for(int i = 0; i < NUM_INPUT; i++){
    printf("Password: %s, Hash:", arr[i].password);
    for(int g = 0; g < 4; g++){
      printf("%u ", arr[i].hash[g]);
    }
    printf("\n");
  }

  
  

  
  
  return 0;
}
