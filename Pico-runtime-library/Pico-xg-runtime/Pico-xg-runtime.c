#include <stdio.h>
#include <string.h>
#include <math.h> 
//#include <stdlib.h>
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "lwip/tcp.h"
#include "xgb_model.h"  

/*
This is a XGBoost model runtime in C for a Raspberry Pi Pico 2 W.  Samples are streamed through a TCP server to the Pico using a ring buffer that the model reads from.
The server sending the samples is a Raspbery PI 5 assumed to be on the same local network as the microcontroller
*/
#define SERVER_IP "10.0.0.224"  //The raspberry PI 5's local IP and port, may change often between runs if the server hasn't turned off DHCP
#define SERVER_PORT 5005

#define BUFFER_SIZE (97 * 2000) //buffer should align to 97 bytes for slight performance gain, also clearly shows that the buffer can hold 2000 samples 
#define FEATURES 24 
#define SAMPLE_SIZE (FEATURES * sizeof(float) + 1) //97 bytes
#define NUM_SAMPLES 2273097

#define LEARNING_RATE 0.1f  
#define OPTIMAL_THRESHOLD 0.5f  
#define BASE_SCORE 0.0f 

//Keep in mind that modifying sample.label with it's name probably doesn't work (not that there is reason to)
typedef struct __attribute__((packed)) { //Make a typedef for convience of making samples on the fly, packed so we don't care about word alignment and get ONLY 97 bytes per sample (not 100).  Doing it like this does force specific byte manipulation which is what memcpy in buffer_read_block does
    float features[FEATURES];
    uint8_t label;
} sample_t;


static uint8_t ring_buffer[BUFFER_SIZE];

static volatile uint32_t write_pos = 0; //Volatile prevents compiler optimizations from messing with these constantly changing values, may not actually be necessary
static volatile uint32_t read_pos  = 0;
static volatile bool connection_closed = false;

static int correct = 0;
static int total = 0;

static struct tcp_pcb* pcb; //Struct tcp_pb manages the TCP connection (Remote IP, Ports, Callbacks etc.), defined in lwip/tcp.h


static uint32_t buffer_available(){ //Returns how many bytes are ready to be read by the model (not samples it has to be >=97) (plz don't overwrite them Mr Pico :( )
    if (write_pos >= read_pos)
        return write_pos - read_pos;
    else
        return BUFFER_SIZE - (read_pos - write_pos);
}

/*
static bool buffer_push(uint8_t byte){ //Writes a received byte to the buffer, returns whether it was successful
    uint32_t next = (write_pos + 1) % BUFFER_SIZE; //the next position will just be the next byte until reaching the end of the buffer where it go back to the beginning 
    //and begins overwriting the previous samples.  Because of this we have to inference on the samples written before they are overrwritten
    if (next == read_pos)
        return false; // would cause overwrriting data (would be caused by some issue with reading samples somehow going slower than writing them, hopefully this doesn't happen)
    ring_buffer[write_pos] = byte;
    write_pos = next;

    return true;
}
*/

static bool buffer_push_block(const uint8_t* data, uint32_t len) { //Writes len bytes to the buffer, more performant than previous implementation as we know the number of byes in a TCP packet, returns if write was successful
    uint32_t free_space; //32 bit memory addresses
    if (write_pos >= read_pos)
        free_space = BUFFER_SIZE - (write_pos - read_pos) - 1; //Opposite of buffer_available
    else
        free_space = read_pos - write_pos - 1;
    if (len > free_space)
        return false;

    uint32_t first_part = BUFFER_SIZE - write_pos;
    if (first_part > len)  //Make sure we don't memcpy out of bounds
        first_part = len;

    memcpy(&ring_buffer[write_pos],data,first_part);
    write_pos =(write_pos + first_part) % BUFFER_SIZE;
    
    //If we reached the end of the buffer, we have to wrap around and memcpy to the start of the buffer. Otherwise we'd either write out of the buffer or misalign the bytes of the samples
    uint32_t remaining = len - first_part; 
    if (remaining > 0){
        memcpy(&ring_buffer[write_pos], data + first_part, remaining);
        write_pos = (write_pos + remaining) % BUFFER_SIZE;
    }
    return true;
}

/*
static bool buffer_pop(uint8_t* byte){ //Reads a byte from the buffer at the given address, returns whether it was successful
    if (read_pos == write_pos) //check here too bc we do not need to overwrite samples PLEASE
        return false; // empty
    *byte = ring_buffer[read_pos];
    read_pos =(read_pos + 1) % BUFFER_SIZE;
    return true;
}
*/

static int buffer_read_block(uint8_t* dest,uint32_t len) { //Same memcpy optimization as before, much faster than looping through each byte, returns whether was successful
    if (buffer_available() < len)
        return -1;
    uint32_t first_part = BUFFER_SIZE - read_pos;
    if (first_part > len)
        first_part = len;

    memcpy(dest,&ring_buffer[read_pos],first_part);
    read_pos = (read_pos + first_part) % BUFFER_SIZE; //Get new read position, will auto wrap due to modulo

    uint32_t remaining = len - first_part;
    if (remaining > 0){ //If the end of the buffer is reached we need to wrap to the beginning
        memcpy(dest + first_part, &ring_buffer[read_pos], remaining);
        read_pos += (read_pos + remaining) % BUFFER_SIZE;
    }
    return 0;
}


float predict_tree(Node* node, const float* X) {  //Returns the leaf value for a specific tree
    int idx = 0;
    while (node[idx].feature != -1) {
        if (X[node[idx].feature] <= node[idx].threshold)
            idx = node[idx].left;
        else
            idx = node[idx].right;

        if (idx == -1) break;  //Safety check
    }
    return node[idx].value;  //return the leaf value
}

// Function to apply sigmoid to the raw score
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));  // exp() is used to compute e^(-x)
}

int classify_sample(const float* X_raw) { //function to classify a single sample
    // Don't have to bother with scaling this time
    float total_score = BASE_SCORE;  // Start with the base score (log-odds)

    //Summing the predictions from all trees
    for (int t = 0; t < N_TREES; t++) {
        total_score += predict_tree(forest[t], X_raw) * LEARNING_RATE;
    }

    float prob = sigmoid(total_score); // Apply sigmoid to get the probability of class 1
    return prob >= OPTIMAL_THRESHOLD ? 1 : 0; // Classify based on the probability (threshold 0.5 currently, may experiment with changes)
}



void process_samples(int* correct, int* total){ //Where the magic happens lwk
    //int* results = malloc(2 * sizeof(int)); //so we can keep track of correct and total samples after the function closes
    while (buffer_available() >= SAMPLE_SIZE){ //Only begin reading a sample if a full sample has been written. Technically could be optimized to start while a sample is still being written but that would likely cause major issues not easily addressable
        sample_t sample;
        buffer_read_block((uint8_t*)&sample,SAMPLE_SIZE); //Reads 97 bytes from the buffer and puts them in the sample struct (using memcpy)
        //From here we are doing normal not direct byte and address manipulation coding on the sample so the packed struct becomes kinda unusable
        //So I copy 96 bytes of the sample (everything but the label) into a float alinged array for the model to then use
        float aligned_features[FEATURES]; 
        memcpy(aligned_features, sample.features, sizeof(aligned_features)); 

        /*
        for (int i=0;i<SAMPLE_SIZE;i++){ //Reads 96 bytes from the buffer byte by byte using a for loop
            buffer_pop((uint8_t*)&sample + i); //This means: pass the base address of the sample casted to uint8_t (so we can go byte by byte rather than increments of sizeof(sample_t)) + the number of bytes alredy written to 
            //you could write &((uint8_t*)&sample)[i] which actually parses to &(*(((uint8_t*)&sample) + i)) which is hilarious so at this much of a low level, just manipulating the virtual addresses directly is genuinely more readable to me
        }
        */
        int pred = classify_sample(aligned_features);
        int true_label = sample.label;
        if(pred == true_label)
            (*correct)++;
        (*total)++;
        //printf("Pred=%d True=%d\n", pred, sample.label);
        //printf("Total: %d",total);
    }
}

static err_t tcp_recv_cb(void* arg, struct tcp_pcb* tpcb, struct pbuf* p ,err_t err){//We have recevied TCP data (yay)
    if (!p){
        printf("Connection closed\n");
        connection_closed = true;
        tcp_close(tpcb);
        return ERR_OK;
    }

    struct pbuf* q = p; //Make a local pbuf address copy just to be safe
    while (q){
        uint8_t* payload = (uint8_t*)q->payload; //Cast the void payload pointer to uint8_t so we can manipulate it byte by byte (you may be able to skip this step if you only use memcpy)
        while (!buffer_push_block(payload, q->len)) {/*q->len is NOT neccecarily 97 bytes for a sample it could be bassically anything (probably 1496 bytes based on debug data),
            we are just writing blocks of data here and it is process_samples()'s job to read 97 byte blocks alinged to make a sample*/
            process_samples(&correct, &total); //buffer full so we should process samples to free space 
        }        
        q = q->next;
    }
    tcp_recved(tpcb, p->tot_len);
    pbuf_free(p);

    return ERR_OK;
}

static err_t tcp_connected(void* arg, struct tcp_pcb* tpcb, err_t err){ //Just confirms it connected properly
    printf("Connected\n");
    tcp_recv(tpcb,tcp_recv_cb);
    /* For refrence, packet buffer this is the struct that actually is received as data from 
    struct pbuf {
        struct pbuf *next; //The next packet in the chain
        void *payload; //Base pointer to the data
        u16_t len; //Length of the data from the base pointer
        u16_t tot_len; //Total length of the TCP chain (I don't use this)
    };
    */
    return ERR_OK;
}

void tcp_client_connect(){ //Connect to the TCP server on our PI 5
    ip_addr_t ip; // Struct defined in "lwip/tcp.h", 32 bytes define an IP address as 4 8 bit ints (ex. 192.168.1.100 -> 0xC0A80164)
    ipaddr_aton(SERVER_IP,&ip); //Convert the IP to a binary ip_addr_t and store it in ip
    pcb = tcp_new(); //Initialize the tcp struct from the top of the file
    tcp_connect(pcb,&ip,SERVER_PORT,tcp_connected); //Start the connection, runs tcp_connected() if it works
}


int main(){
    stdio_init_all();
    if (cyw43_arch_init())
        return -1;

    cyw43_arch_enable_sta_mode();
    while (true) {
        printf("Connecting to Wi-Fi...\n");
        int result = cyw43_arch_wifi_connect_timeout_ms("BP205","BeaverPlaza205!",CYW43_AUTH_WPA2_AES_PSK,30000);
        if (result == 0) {
            printf("Connected.\n");
            break;
        }
        printf("Retrying...\n");
        sleep_ms(5000);
    }
    printf("Connected to BP205.\n");
    // Read the ip address in an at all readable way
    uint8_t* ip_address = (uint8_t*)&(cyw43_state.netif[0].ip_addr.addr);
    printf("IP address %d.%d.%d.%d\n", ip_address[0], ip_address[1], ip_address[2], ip_address[3]);
    //Connect to PI 5
    tcp_client_connect();

    printf("polling tcp\n");
    while(true) {
        
        cyw43_arch_poll();
        //printf("done with that\n");
        process_samples(&correct,&total);
        printf("total: %d  correct: %d\n",total,correct);
        if(total == NUM_SAMPLES)
            break;
        if (connection_closed) {
            process_samples(&correct,&total);
            if (buffer_available() < SAMPLE_SIZE)
                break;
        }
        sleep_ms(10);
    }

    float accuracy = (float)correct / total;
    printf("Total samples: %d\n", total);
    printf("Correct: %d\n", correct);
    printf("Accuracy: %.4f (%.2f%%)\n", accuracy, accuracy * 100.0f);
    
    return 0;
}