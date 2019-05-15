#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"
#include "socket.h"
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

/*these denote whether or not the hash table is at capacity
 *in a certain bin in a given addToHashTable call*/
#define SUCCESS 43
#define FAILURE 21

#define HASH_LENGTH 32

/////////////////////////////// GLOBALS AND DATA STRUCTURES  /////////////////////////////////

__device__ size_t POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};

__device__ int num_cracked = 0;

typedef struct hashInfo{
  char  password[7];
  uint  hash[4];
  int empty = 1; //denotes whether a hashInfo_t has a hash value or not
  int length; //gives the number of items in a bin of a hash table
}hashInfo_t;

/////////////////////////////// GPU FUNCTIONS  /////////////////////////////////

/*
  Determines if the hash dentoed by hash exists in the hash table denoted by table. 
  If it does, the function returns 1, otherwise it returns 0
  Password signifes the word to be inserted into the hash table if it matches
*/
__device__ int isHash(hashInfo_t * table, uint * hash, char * password){
  //get the bin of the hash by taking the first byte of the first uint of the hash
  unsigned char byte;
  byte = (hash[0]&0xFF);

  //loops through the bin -- first element of the bin is empty but denotes the size of the bin
  for(int i = 1; i < table[DEPTH*byte].length+1; i++){
    int index = DEPTH*byte+i;
    if(!table[index].empty
       && table[index].hash[0] == hash[0]
       && table[index].hash[1] == hash[1]
       && table[index].hash[2] == hash[2]
       && table[index].hash[3] == hash[3]){
      num_cracked++;
      //copies password to the password field of the hashInfo_t with the matching hash
      memcpy(table[index].password, password, PASSWORD_LENGTH*sizeof(char));
      return 1;
       
    }
     
  }
  return 0;
}

/*
  The kernel, crack, runs on the gpu and brute force cracks 6 character, alphabetic passwords
  hashData denotes a hash table
*/
__global__  void crack(hashInfo_t * hashData){
  
  //get string permuation
  size_t tempNum =((size_t) blockIdx.x) * ((size_t) NUM_THREADS) +((size_t) threadIdx.x);

  //starts at the lowest value
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

  //checks for matching hash and inserts word into hashData if candidate_hash matches
  isHash(hashData, candidate_hash, word);
  return;
    
}

///////////////////////// HASH TABLE FUNCTIONS ////////////////////////////////

//add hash to hash table
//uses first byte of hash to index hash in hashTable
//returns SUCCESS if there is space in the bin denoted by the first byte for the hash
//returns FAILURE otherwise
int addToTable(hashInfo_t * table, uint * hash){
  //get the bin of the hash by taking the first byte of the first uint of the hash
  unsigned char byte;
  byte = (hash[0]&0xFF);

  //handles insertion and length incrementing in the bin given by byte
  if(table[DEPTH*byte].length == 0){
    table[DEPTH*byte+1].hash[0] = hash[0];
    table[DEPTH*byte+1].hash[1] = hash[1];
    table[DEPTH*byte+1].hash[2] = hash[2];
    table[DEPTH*byte+1].hash[3] = hash[3];
         
    table[DEPTH*byte+1].empty = 0;
    table[DEPTH*byte].length++;
    return SUCCESS;
  }

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
 
  return FAILURE;
}

/////////////////////////// MAIN ////////////////////////////

int main(int argc, char* argv[]){
  // Make sure the arguments include a port
  if(argc !=3 ) {
    fprintf(stderr, "Usage: %s <port number>\n", argv[0]);
    exit(1);
  }
  char * name = argv[1];
  unsigned short server_port = atoi(argv[2]);

  //calculates number of blocks needed assuming every thread computes one
  //possible six-character, alpabetic string permuation.
  int number_of_blocks = (SIXTH_POWER+NUM_THREADS)/NUM_THREADS;

  
  //Connect to server
  int server_socket_fd = socket_connect(name, server_port);
  if(server_socket_fd == -1){
    perror("Connection to server failed.\n");
    exit(2);
  }
  usleep(10000000);
  printf("Connected\n");

  //read in input sent over from the host maching until the the host machine terminates the
  //connection or the read fucntion errors
  int rc;
  hashInfo_t* hash_table = (hashInfo*) malloc(sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH);
  while((rc = read(server_socket_fd, hash_table, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH)) <= 0){
 
     

    //Create the data structure to pass to the GPU
    hashInfo_t  * gpu_hashTable;
    if(cudaMalloc(&gpu_hashTable, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH) != cudaSuccess){
      perror("Cuda Malloc Failed\n");
      exit(2);
    }

    //Copy over our provided hashes in arr to the GPU_arr for analysis.
    if(cudaMemcpy(gpu_hashTable, hash_table, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH,
                  cudaMemcpyHostToDevice) != cudaSuccess){
      close(server_socket_fd);
      perror("Drone CPU to GPU memcpy Failed\n");
      exit(2);
    }

    printf("Cracking...\n");

    crack<<<number_of_blocks, NUM_THREADS>>>(gpu_hashTable);

    //Ensure all CUDA threads have terminated
    if(cudaDeviceSynchronize() != cudaSuccess){
      close(server_socket_fd);
      perror("Drone CUDA Thread Synchronization Error\n");
      exit(2);
    }
    printf("Done\n");

    //Copy back the cracked passwords from the GPU.
    if(cudaMemcpy(hash_table, gpu_hashTable, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH,
                  cudaMemcpyDeviceToHost) != cudaSuccess){
      close(server_socket_fd);
      perror("Cuda GPU to CPU memcpy Failed\n");
      exit(2);
    }
        
    //Send the hash table back to the server
    if(write(server_socket_fd, hash_table, sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH) <= 0){
      perror("Write Passback Failed\n");
      close(server_socket_fd);
      exit(2);
    }

    printf("Waiting for another hash table.\n");
    free(hash_table);
    hash_table = (hashInfo*) malloc(sizeof(hashInfo_t) * NUMBER_OF_BINS * DEPTH);
  }

  if(rc == 0){
    printf("Connection closed\n");
  }
  else{
    perror("Drone could not read hash table.\n");
  }
   
  
  close(server_socket_fd);
  return 0;
}
