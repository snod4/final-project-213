#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"
#include <unistd.h>

#define NUM_THREADS 128
#define MAX_USERNAME_LENGTH 64
#define DEPTH 100
#define PASSWORD_LENGTH 6
#define SIXTH_POWER  (26 * 26 * 26 * 26 * 26* 26)
#define FIFTH_POWER (26 * 26 * 26 * 26 * 26)
#define FOURTH_POWER (26 * 26 * 26 * 26)
#define THIRD_POWER (26 * 26 * 26)
#define SECOND_POWER (26 * 26)
#define FIRST_POWER 26
#define NUMBER_OF_BINS 256

//these denote whether or not the hash table is at capacity in a certain bin in a given addToHashTable call
#define SUCCESS 43
#define FAILURE 21

#define HASH_LENGTH 32

__device__ size_t POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};

__device__ int num_cracked = 0;
typedef struct hashInfo{
  char  password[7];
  uint  hash[4];
  int empty = 1;
  int length;
}hashInfo_t;


__device__ int isHash(hashInfo_t * table, uint * hash, char * password){
  unsigned char byte;
  byte = (hash[0]&0xFF);
  for(int i = 1; i < table[DEPTH*byte].length+1; i++){
    int index = DEPTH*byte+i;
    if(!table[index].empty
       && table[index].hash[0] == hash[0]
       && table[index].hash[1] == hash[1]
       && table[index].hash[2] == hash[2]
       && table[index].hash[3] == hash[3]){
      num_cracked++;
      memcpy(table[index].password, password, PASSWORD_LENGTH*sizeof(char));
      return 1;
       
    }
     
  }
  return 0;
}

/* __device__ void printGPUTable(hashInfo_t * table){ */
/*   for(int r = 0; r < NUMBER_OF_BINS; r++){ */
/*     for( int c = 0; c < DEPTH; c++){ */
/*       if(!table[DEPTH*r+c].empty){ */
/*         printf("Password: %s, Hash: ",table[DEPTH*r+c].password); */
/*         for(int g = 0; g < 4; g++){ */
/*           printf("%u ", table[DEPTH*r+c].hash[g]); */
/*         } */
/*         printf("\n"); */
/*       } */
/*     } */
/*   } */
/* } */

/* __global__ void test(hashInfo_t * table){ */
/*   printGPUTable(table); */
/* } */
//Crack is a function that runs on the GPU and brute forces given hashes.

__global__  void crack(hashInfo_t * hashData){
  
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
  isHash(hashData, candidate_hash, word);
  // printf("Got out\n");
  return;
    
}

//add hash to hash table
//uses first byte of hash to index hash in hashTable
//returns SUCCESS if there is space in the bin denoted by the first byte for the hash
//returns FAILURE otherwise
int addToTable(hashInfo_t * table, uint * hash){
  unsigned char byte;
  byte = (hash[0]&0xFF);
 
  if(table[DEPTH*byte].length == 0){
      table[DEPTH*byte+1].hash[0] = hash[0];
      table[DEPTH*byte+1].hash[1] = hash[1];
      table[DEPTH*byte+1].hash[2] = hash[2];
      table[DEPTH*byte+1].hash[3] = hash[3];
         
      table[DEPTH*byte+1].empty = 0;
      table[DEPTH*byte].length++;
      return SUCCESS;
  }
  
  /* for(int i = 1; i < table[DEPTH*byte].length+2; i++){ */
    
  /*    int index = DEPTH*byte+i; */
  /*    //printf("%d\n",table[index].empty); */
  /*   if(table[index].empty){ */
      
  /*     table[index].hash[0] = hash[0]; */
  /*     table[index].hash[1] = hash[1]; */
  /*     table[index].hash[2] = hash[2]; */
  /*     table[index].hash[3] = hash[3]; */
         
  /*     table[index].empty = 0; */
  /*     table[DEPTH*byte].length++; */
  /*     // printf("Add Success\n"); */
  /*     return SUCCESS; */
  /*   } */
  /* } */

  int placement = table[DEPTH*byte].length+1;
  if(placement - 1 < DEPTH  - 1){
    table[DEPTH*byte+placement].hash[0] = hash[0];
    table[DEPTH*byte+placement].hash[1] = hash[1];
    table[DEPTH*byte+placement].hash[2] = hash[2];
    table[DEPTH*byte+placement].hash[3] = hash[3];
    table[DEPTH*byte].length++;
    table[DEPTH*byte+placement].empty = 0;
    return SUCCESS;
    }
  // printf("Add FAILURE\n");
  return FAILURE;
}


void printHashTable(hashInfo_t  * table){
  int count = 0;
  for(int r = 0; r < NUMBER_OF_BINS; r++){
    for( int c = 1; c < table[r*DEPTH].length + 1; c++){
      printf("Password: %s, Hash: ",table[DEPTH*r+c].password);
      for(int g = 0; g < 4; g++){
        printf("%u ", table[DEPTH*r+c].hash[g]);
      }
      count++;
      printf("Bin: %d Count: %d\n",r, count);
    }
  }

}



void initializeTable(hashInfo_t * table){
  for(int i = 0; i < NUMBER_OF_BINS*DEPTH; i++){
    table[i].empty = 1;
    table[i].length = 0;
  }
  
}

int sumTable(hashInfo_t * table){
  int sum = 0;
  for(int i = 0; i < NUMBER_OF_BINS; i++){
    sum+= table[i*DEPTH].length;
  }
  return sum;
}


int main(int argv, char* args[]){
  printf("H\n");
  printf("E\n");
  
  hashInfo_t  * gpu_hashTable;
  int counter = 0;

      
      
  FILE * file = fopen("outputFile.txt", "r");
  printf("OPENED FILE\n");
  int NUM_INPUT = 0;
  fscanf(file, "%d", &NUM_INPUT);
  printf("Number of Input: %d\n", NUM_INPUT);
  
  int number_of_blocks = (308915776+NUM_THREADS)/NUM_THREADS;
  //  hashInfo_t arr[NUM_INPUT];
  
  
  uint hash[4];
  char trash_can[7];

  //Grab the input hashes from a file specified by the user in argv[1].{
  while(counter < NUM_INPUT){
   hashInfo_t *  hashTable = (hashInfo_t *)malloc(sizeof(hashInfo_t)*NUMBER_OF_BINS * DEPTH);
   initializeTable(hashTable);
    if(counter != 0){
      addToTable(hashTable,hash);
      counter++;
      // printf("Added Hash: %d\n", counter);
    }
    printf("Counter: %d\n", counter);
    while(fscanf(file, "%s", trash_can) != EOF){
      for(int i = 0; i < 4; i++){
        fscanf(file, "%u", &hash[i]);
      }

      if(addToTable(hashTable, hash) == FAILURE){
        break;
      }
      counter++;
      // printf("Added Hash: %d\n", counter);
    }

    //Create the data structure to pass to the GPU
 
    if(cudaMalloc(&gpu_hashTable, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH) != cudaSuccess){
      perror("Cuda Malloc Failed\n");
      exit(2);
    }

    //Copy over our provided hashes in arr to the GPU_arr for analysis.
    if(cudaMemcpy(gpu_hashTable, hashTable, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH, cudaMemcpyHostToDevice) != cudaSuccess){
      perror("Cuda CPU to GPU memcpy Failed\n");
      exit(2);
    }

    //Crack the provided hashes on the GPU
    printf("Begin Portion of Cracking\n");
    
    crack<<<number_of_blocks, NUM_THREADS>>>(gpu_hashTable);
     

    //Ensure all CUDA threads have terminated
    if(cudaDeviceSynchronize() != cudaSuccess){
      perror("CUDA Thread Synchronization Error\n");
      exit(2);
    }
    printf("End of Portion\n");

    //Copy back the cracked passwords from the GPU.
    if(cudaMemcpy(hashTable, gpu_hashTable, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH, cudaMemcpyDeviceToHost) != cudaSuccess){
      perror("Cuda GPU to CPU memcpy Failed\n");
      exit(2);
    }

    //Print the cracked passwords. Eventually we should delete this and automate
    //password cracking sucess when we scale up the amount of passwords to crack
    printHashTable(hashTable);

    printf("Num Of Items: %d\n", sumTable(hashTable));
    free(hashTable);
  }
  
  

  
  
  return 0;
}
