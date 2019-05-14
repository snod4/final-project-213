#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"
#include "socket.h"
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 128
#define MAX_USERNAME_LENGTH 64
#define DEPTH 100
#define PASSWORD_LENGTH 6
#define MIN_DISTRIBUTED_PASSWORDS 3600

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

////////////////////////////// GLOBALS AND DATA STRUCTURES  /////////////////////////////////

//lock for reading the file
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int sever_socket_fd;

__device__ size_t POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};

__device__ int num_cracked = 0;

//struct for packaging the arguments to pass to the threads
struct args{
  FILE * file;
  int client_socket_fd;
}args_t;

//struct for the list of threads -- not necessary in current implementation
struct threadList{
  threadNode *  head;
}threadList_t;

//node in the list of threads
struct threadNode{
  pthread_t thread;
  struct threadNode * next;
}threadNode_t;


typedef struct hashInfo{
  char  password[7];
  uint  hash[4];
  int empty = 1;
  int length;
}hashInfo_t;

/////////////////////////// THREAD FUNCTIONS /////////////////////////////////

/*
  adds a thread into the threadList_t list
*/
void addThread(pthread_t thread, threadList_t * list){
  pthread_t * t = (pthread_t *) malloc(sizeof(pthread_t)); 
  t->thread = thread;
  t->next = NULL;
  threadNode * node = list->head;
  if(list->head == NULL){
    list->head = t;
  }
  else{
    t->next = list->head;
    list->head = t;
  }
  
}

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

  //Checks for matching hash and inserts word into hashData if candidate_hash matches
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

/*
  Prints the contents of the hash table in the order they 
  are stored in the hash table, which is just an array.
  It does not preserve the order of the hashes in the outputFile.txt.
*/
void printHashTable(hashInfo_t  * table){
  //This nested loop takes into account that all bins the hash table are not filled
  //and, as such, avoids the empty bins
  for(int r = 0; r < NUMBER_OF_BINS; r++){
    for( int c = 1; c < table[r*DEPTH].length + 1; c++){
      printf("Password: %s, Hash: ",table[DEPTH*r+c].password);
      for(int g = 0; g < 4; g++){
        printf("%u ", table[DEPTH*r+c].hash[g]);
      }
      printf("\n");
    }
  }
}


/*
  Initializes the first element of each bin, which holds the number of items in each bin given by length
*/
void initializeTable(hashInfo_t * table){
  for(int i = 0; i < NUMBER_OF_BINS*DEPTH; i++){
    table[i].empty = 1; //true
    table[i].length = 0; //nothing in the bin
  }
  
}

/*
  handles cracking on the local machine in its own thread
 */
void * local_crack(void * args){

  FILE * file = ((args_t *) args)->file;
  
  //calculates number of blocks needed assuming every thread computes one
  //possible six-character, alpabetic string permuation.
  int number_of_blocks = (SIXTH_POWER+NUM_THREADS)/NUM_THREADS;

  //gets hash from file
  uint hash[4];
  //gets password which is also stored in file but is not needed for this code -- ignores it
  char trash_can[7];
  //value returned by fscanf
  int eof = 0;

  //Grab the input hashes from a file ideally specified by the user but, for our purposes, specified by outputFile.txt instead
  //If one of the bins of the hash table fills ups completely, the items in the table must be processed. This condition ensures
  //that even if it takes multiple passes, all items file will be processed
  while(eof != EOF){
    
    //allocates space for the hash table
    hashInfo_t *  hashTable = (hashInfo_t *)malloc(sizeof(hashInfo_t)*NUMBER_OF_BINS * DEPTH);
    initializeTable(hashTable);

    //Processes previous item if previous addToTable call returned FAILURE
    if(counter != 0){
      addToTable(hashTable,hash);
    }

    //locks file for reading to avoid double counting potentially
    pthread_mutex_lock(&lock);
    //Reads in input, ignoring passwords and storing hashes
    while((eof = fscanf(file, "%s", trash_can)) != EOF){
      for(int i = 0; i < 4; i++){
        fscanf(file, "%u", &hash[i]);
      }
      
      //This condition triggers when one of the bins in the hash table is full
      if(addToTable(hashTable, hash) == FAILURE){
        break;
      }
    }
    pthread_mutex_unlock(&lock);

    //Create the data structure to pass to the GPU
    hashInfo_t  * gpu_hashTable;
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

    //Print the cracked passwords.
    printHashTable(hashTable);

    //frees hashTable for this portion of the cracking process
    free(hashTable);
  }
}

/*
  Listens for incoming connections from other machines
*/
void incoming_crack(FILE * file){
  
  
  hashInfo_t  * gpu_hashTable;
  int counter = 0;
  
  int number_of_blocks = (308915776+NUM_THREADS)/NUM_THREADS;
  uint hash[4];
  char trash_can[7];
  
  int NUM_INPUT = 0;
  fscanf(file, "%d", &NUM_INPUT);
  printf("Number of Input: %d\n", NUM_INPUT);
  
  unsigned short port = 0;
  
  // open socket
  server_socket_fd = server_socket_open(&port);
  if(server_socket_fd == -1) {
    perror("Server socket was not opened");
    exit(2);
  }
  
  printf("Server Socket ID: %d\n",server_socket_fd);

  if(listen(server_socket_fd, 10)) {
    perror("listen failed\n");
    exit(2);
  }
  
  //defines threadList and thread to create
  threadList_t threadList;
  pthread_t thread;
  
  args_t * args;
  //listens for incoming connections
  while(1){
    int client_socket_fd = server_socket_accept(server_socket_fd);
    if(client_socket_fd == -1) {
      perror("accept failed");
      exit(2);
    }
    //packages file into args
    args = (args_t *)malloc(sizeof(args_t));
    args->file = file;
    args->client_socket_fd = client_socket_fd;

    //create the thread
    pthread_create(&thread, NULL, distribute_crack, args);
    //add thread to the threadList
    addThread(thread, threadList);
  }
  
}

/*
  Sends data from outputFile.txt to connected machine running crackDrone
*/
void * distribute_crack(void * arg){

  //holds hash
  uint hash[4];
  //exists in order to get past the password in the outputFile.txt
  char trash_can[7];

  //gets file from arg
  FILE * file =((args_t *) arg)->file;
  //gets client_socket_fd from arg
  int client_socket_fd = ((arg_t *)arg)->client_socket_fd;

  //counts number of items read in
  int counter;
  //value returned by fscanf
  int eof; = 0;

  //Grab the input hashes from a file ideally specified by the user but, for our purposes, specified by outputFile.txt instead
  //If one of the bins of the hash table fills ups completely, the items in the table must be processed. This condition ensures
  //that even if it takes multiple passes, all items in file will be processed
  while(eof != EOF){

     //allocates space for the hash table
    hashInfo_t *  hashTable = (hashInfo_t *)malloc(sizeof(hashInfo_t)*NUMBER_OF_BINS * DEPTH);
    initializeTable(hashTable);

    //Processes previous item if previous addToTable call returned FAILURE
    if(counter != 0){
      addToTable(hashTable,hash);
      counter++;
    }
    //locks file for reading to avoid double counting potentially
    pthread_mutex_lock(&lock);
    while((eof =fscanf(file, "%s", trash_can)) != EOF){
      for(int i = 0; i < 4; i++){
        fscanf(file, "%u", &hash[i]);
      }
      
      if(addToTable(hashTable, hash) == FAILURE){
        break;
      }
      counter++;
    }
    pthread_mutex_unlock(&lock);

    //sends hash table to the connected machine running crackDrone
    if(write(client_socket_fd, hashTable, sizeof(hashInfo_t) * DEPTH * NUMBER_OF_BINS) <= 0){
      perror("Write Pass-To failed\n");
      close(client_socket_fd);
      exit(2);
    }
    //reads back crack passwords from connected machine
    int error;
    if((error = read(client_socket_fd, hashTable, sizeof(hashInfo_t) * DEPTH * NUMBER_OF_BINS)) <= 0){
      if(error == 0){
        printf("Conection Closed\n");
      }
      else{
      perror("Read Pass-Back failed\n");
      }
      close(client_socket_fd);
      exit(2);
    }
    //prints cracked passwords along with hashes
    printHashTable(hashTable);
  }

  
  
  close(client_socket_fd);
}


int main(int argv, char* args[]){
 
 
  //opens outputFile.txt -- could be modified to be any file    
  FILE * file = fopen("outputFile.txt", "r");
  printf("OPENED FILE\n");

  //gets number of input from file
  int NUM_INPUT;
  fscanf(file, "%d", &NUM_INPUT);
  
  //packages file to pass to local_crack
  args_t args;
  args.file = file;
  
  //creates thread for local_crack
  pthread_t thread;
  pthread_create(&thread, NULL, local_crack, args);

  //does not distribute the file data unless the number of input exceeds MIN_DISTRIBUTED_PASSWORDS
  //which takes about one second to crack on the classroom machines
  if(NUM_INPUT > MIN_DISTRIBUTED_PASSWORDS){
    incoming_crack(file);
  }

  return 0;
}
