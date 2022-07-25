#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/** Width of model input image */
#define WIDTH                           (44U)

/** Height of model input image */
#define HEIGHT                          (73U)

/** Model input image frame size */
#define FRAME_SIZE                      (WIDTH * HEIGHT)

/** Number of trained model parameters */
#define MODEL_WEIGHTS                   (253U)

/** Conv2D 5*5 Matrix kernel weights */
#define KERNEL_WEIGHTS                  (25U)

/** Size of trained weights in dense layer after Flattened */
#define DENSE_WEIGHTS                   (120U)

/** Width of model image at Layer 2 */
#define L2_WIDTH                        (40U)

/** Height of model image at Layer 2 */
#define L2_HEIGHT                       (69U)

/** Width of model image at Layer 3 */
#define L3_WIDTH                        (36U)

/** Height of model image at Layer 3 */
#define L3_HEIGHT                       (65U)

/** Width of model image at Layer 4 */
#define L4_WIDTH                        (32U)

/** Height of model image at Layer 4 */
#define L4_HEIGHT                       (61U)

/** Width of model image at Layer 5 */
#define L5_WIDTH                        (28U)

/** Height of model image at Layer 5 */
#define L5_HEIGHT                       (57U)

/** Width of model image at Maxpool Layer */
#define MAXPOOL_WIDTH                   (24U)

/** Width of model image at Maxpool Layer */
#define MAXPOOL_HEIGHT                  (53U)

/** Width of model image after Maxpool Layer  */
#define DENSE_WIDTH                     (12U)

/** Width of model image after Maxpool Layer */
#define DENSE_HEIGHT                    (10U)

/** 5X5 Matrix to store pixel data and kernel matrices */
typedef float Mat5[5][5];

/** 5X5 Matrix to store pixel data and kernel matrices */
typedef float MatMaxPool5X2[5][2];

/** Creating Global Place holders for trained model weights*/
Mat5 Kernel_L1, Kernel_L2, Kernel_L3, Kernel_L4, Kernel_L5;
float Bais_L1, Bais_L2, Bais_L3, Bais_L4, Bais_L5, Bais_L7, Bais_L8;
float* Weights_L7 = (float*)malloc(DENSE_WEIGHTS * sizeof(float));
float Weight_L8;

/**
 * @brief Function to read 5*5 kernel form trained weights
 * 
 * @param[in] src           Source data pointer to read model weights
 * @param[inout] lReturn    Kernel with model weights 
 */
void Populate_KernelMat5(float* src, Mat5 lReturn);

/**
 * @brief Function to read dense layer weights form trained weights
 * 
 * @param[in] src       Source data pointer to read model weights
 * @param[inout] dst    Destination data pointer to write model weights
 * @param[in] size      Length of dense weights
 */
void Populate_DenseWeights(float* src, float* dst, int size);

/**
 * @brief Extracts pixel data from source image of shape 5*5 pixels
 * 
 * @param[in] row           Pixel row position to copy pixel data from
 * @param[in] column        Pixel column position to copy pixel data from
 * @param[in] width         Width of source image  
 * @param[in] src           Source image pointer from which pixel data is required 
 * @param[inout] lReturn    Mat5 filled with source image pixel data 
 */
static void Populate_PixelMat5(int row, int column, int width, float* src, Mat5 lReturn);

/**
 * @brief Performs Conv2D operation for Con2D Layers from tensorflow
 * 
 * @param[in] src       Source image pointer on which Conv2D is performed
 * @param[in] height    Height of source image
 * @param[in] width     Width of source image
 * @param[inout] dst    Destination image pointer on which pixel data is stored obtained after Conv2D operation    
 * @param[in] kernel    Kernl values to operate Conv2D     
 */
void Conv2DLayer_KernelSize5(float* src, int height, int width, float* dst, Mat5 kernel);

/**
 * @brief Adds bais values to datas obtained after convolution and fully connected layer and also performs ELU activation function
 * 
 * @param[in] val_bais  Bais value to add on obtained values 
 * @param[in] src_size  Size of data fed to add bais values
 * @param[inout] src    Pointer to source data
 */
void AddBais(float val_bais, int src_size, float* src);

/**
 * @brief Gets maximum values of the matrix for Maxpool layer
 * 
 * @param[in] mat_pixel Pixel values to get maximum value 
 * @return float        Maximum value among 5*2 matrix
 */
float Get_Maximum(MatMaxPool5X2 mat_pixel);

/**
 * @brief Performs Maxpooling operation with stride_width and stride_height
 * 
 * @param[in] src           Source data pointer to perform Maxpooling 
 * @param[inout] dst        Destination data pointer to store data after performing Maxpooling 
 * @param[in] stride_width  Maxpool stride witdh size 
 * @param[in] stride_height Maxpool stride height size
 * @param[in] in_height     Height of 2D input data
 * @param[in] in_width      Width of 2D input data
 * @param[in] dst_width     Width of 2D output data 
 */
void MaxPoolLayer(float* src, float* dst, int stride_width, int stride_height, int in_height, int in_width, int dst_width);

/**
 * @brief Performs dense calculation for fully connected layer. Output value after dense calculation for input depth of 1
 * 
 * @param[in] src_pixel     Source data pointer to perform dense calculation 
 * @param[in] kernel        Trained model weights to perform calculation
 * @param[in] length        Input dense layer size
 * @return float            Return value for fully connected value
 */
float FullyConnectedLayer(float* src_pixel, float* kernel, int length);

/**
 * @brief   Performs Exponential Linear Unit operation 
 * 
 * @param[inout] src_pixel  Input to calculate ELU
 * @param[in] length        Number of input data values
 */
void Activation_ELU(float* src_pixel, int length);

/**
 * @brief Performs sigmoid operation
 * 
 * @param[in] src_pixel Input to calculate sigmoid 
 * @param[in] length    Number of input data values
 */
void Activation_Sigmoid(float* src_pixel, int length);

/**
 * @brief Read trained model parameters and stores in appropriate global place holders 
 * 
 * @param[in] fname Filename to read trained model parameters 
 */
void Read_ModelParamaters(char* fname);

/**
 * @brief Port tensorflow keras model to C Function 
 * 
 * @param[in] src_pixel Source data pointer for model input image 
 * @return float        Prediction value for binary clasification (Yes/No - 1/0)
 */
float Model_Inference(float* src_pixel);


int main(int argc, char* argv[])
{
    char filename[] = "weights.bin";
    int valid = 0;
    int invalid = 0;

    Read_ModelParamaters(filename);

    for (int i = 0; i < 50; i++)
    {
        char path[] = "E:/read_write/CNPY/test_data_no/";
        char name[10] = "data";
        char ext[5] = ".bin";
        sprintf(name, "%d", i);
        strcat(name, ext);
        char* fullpath = (char*)calloc((strlen(path) + strlen(name) + 1), sizeof(char));
        strcat(fullpath, path);
        strcat(fullpath, name);

        FILE* fp_testdata;
        fp_testdata = fopen(fullpath, "rb");
        float* audio = (float*)malloc(FRAME_SIZE * sizeof(float));
        fread(audio, sizeof(float), FRAME_SIZE, fp_testdata);
        fclose(fp_testdata);

        float prediction = Model_Inference(audio);

        if (prediction > 0.5)
            invalid++;
        else
            valid++;

        free(audio);
    }

    printf("Valid: %d, Invalid: %d, Total: %d \n", valid, invalid, valid + invalid);

    valid = 0;
    invalid = 0;

    for (int i = 0; i < 50; i++)
    {
        char path[] = "E:/read_write/CNPY/test_data_yes/";
        char name[10] = "data";
        char ext[5] = ".bin";
        sprintf(name, "%d", i);
        strcat(name, ext);
        char* fullpath = (char*)calloc((strlen(path) + strlen(name) + 1), sizeof(char));
        strcat(fullpath, path);
        strcat(fullpath, name);

        FILE* fp_testdata;
        fp_testdata = fopen(fullpath, "rb");
        float* audio = (float*)malloc(FRAME_SIZE * sizeof(float));
        fread(audio, sizeof(float), FRAME_SIZE, fp_testdata);
        fclose(fp_testdata);

        float prediction = Model_Inference(audio);

        if (prediction > 0.5)
            valid++;
        else
            invalid++;

        free(audio);
    }

    printf("Valid: %d, Invalid: %d, Total: %d \n", valid, invalid, valid + invalid);

    return 0;
}

void Read_ModelParamaters(char* fname)
{
    FILE* fp_weights;
    fp_weights = fopen(fname, "rb");
    float* ModelWeights = (float*)malloc(MODEL_WEIGHTS * sizeof(float));
    fread(ModelWeights, sizeof(float), MODEL_WEIGHTS, fp_weights);
    fclose(fp_weights);

    Populate_KernelMat5(ModelWeights, Kernel_L1);
    Bais_L1 = *(ModelWeights + 25);
    Populate_KernelMat5(ModelWeights + 26, Kernel_L2);
    Bais_L2 = *(ModelWeights + 51);
    Populate_KernelMat5(ModelWeights + 52, Kernel_L3);
    Bais_L3 = *(ModelWeights + 77);
    Populate_KernelMat5(ModelWeights + 78, Kernel_L4);
    Bais_L4 = *(ModelWeights + 103);
    Populate_KernelMat5(ModelWeights + 104, Kernel_L5);
    Bais_L5 = *(ModelWeights + 129);
    Populate_DenseWeights(ModelWeights + 130, Weights_L7, DENSE_WEIGHTS);
    Bais_L7 = *(ModelWeights + 250);
    Weight_L8 = *(ModelWeights + 251);
    Bais_L8 = *(ModelWeights + 252);
}


float Model_Inference(float* src_pixel)
{
    /* ML Audio Classifier Begins */
    /*Layer 1*/
    float* audio_out_layer1 = (float*)malloc(L2_WIDTH * L2_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(src_pixel, HEIGHT, WIDTH, audio_out_layer1, Kernel_L1);
    AddBais(Bais_L1, L2_WIDTH * L2_HEIGHT, audio_out_layer1);
    Activation_ELU(audio_out_layer1, L2_WIDTH * L2_HEIGHT);

    /*Layer 2*/
    float* audio_out_layer2 = (float*)malloc(L3_WIDTH * L3_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer1, L2_HEIGHT, L2_WIDTH, audio_out_layer2, Kernel_L2);
    AddBais(Bais_L2, L3_WIDTH * L3_HEIGHT, audio_out_layer2);
    Activation_ELU(audio_out_layer2, L3_WIDTH * L3_HEIGHT);

    /*Layer 3*/
    float* audio_out_layer3 = (float*)malloc(L4_WIDTH * L4_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer2, L3_HEIGHT, L3_WIDTH, audio_out_layer3, Kernel_L3);
    AddBais(Bais_L3, L4_WIDTH * L4_HEIGHT, audio_out_layer3);
    Activation_ELU(audio_out_layer3, L4_WIDTH * L4_HEIGHT);

    /*Layer 4*/
    float* audio_out_layer4 = (float*)malloc(L5_WIDTH * L5_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer3, L4_HEIGHT, L4_WIDTH, audio_out_layer4, Kernel_L4);
    AddBais(Bais_L4, L5_WIDTH * L5_HEIGHT, audio_out_layer4);
    Activation_ELU(audio_out_layer4, L5_WIDTH * L5_HEIGHT);

    /*Layer 5*/
    float* audio_out_layer5 = (float*)malloc(MAXPOOL_WIDTH * MAXPOOL_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer4, L5_HEIGHT, L5_WIDTH, audio_out_layer5, Kernel_L5);
    AddBais(Bais_L5, MAXPOOL_WIDTH * MAXPOOL_HEIGHT, audio_out_layer5);
    Activation_ELU(audio_out_layer5, MAXPOOL_WIDTH * MAXPOOL_HEIGHT);

    /*Layer 6 - MaxPooling of size (5,2)*/
    float* audio_out_layer6 = (float*)malloc(DENSE_WIDTH * DENSE_HEIGHT * sizeof(float));
    MaxPoolLayer(audio_out_layer5, audio_out_layer6, 2, 5, MAXPOOL_HEIGHT, MAXPOOL_WIDTH, DENSE_WIDTH);

    /* Layer 7 Dense layer calculations and Add baises */
    float temp = FullyConnectedLayer(audio_out_layer6, Weights_L7, DENSE_WEIGHTS);
    AddBais(Bais_L7, 1, &temp);
    Activation_ELU(&temp, 1);

    /* Layer 8 */
    float retval = FullyConnectedLayer(&temp, &Weight_L8, 1);
    AddBais(Bais_L8, 1, &retval);
    Activation_Sigmoid(&retval, 1);
    
    return retval;
}

void AddBais(float val_bais, int size, float* src)
{
    int iter = 0;
    do
    {
        *(src + iter) += val_bais;
        iter++;
    } while (iter < size);
}

void Activation_ELU(float* src_pixel, int length)
{
    int i = 0;
    do
    {
        if (*(src_pixel + i) < 0)
            *(src_pixel + i) = (float)exp(*(src_pixel + i)) - 1;
        i++;
    } while (i < length);
}

void Activation_Sigmoid(float* src_pixel, int length)
{
    int i = 0;
    do 
    {
        *(src_pixel + i) = (float)1 / (1 + exp(- *(src_pixel + i)));
        i++;

    } while(i < length);
}

void Populate_KernelMat5(float* src, Mat5 lReturn)
{
    int i, j, position;

    for (i = 0; i < 5; i++)
    {
        for (j = 0; j < 5; j++)
        {
            position = i * 5 + j;
            lReturn[i][j] = *(src + position);
        }
    }
}/* End of function Populate_KernelMat5 */

void Populate_PixelMat5(int row, int column, int width, float* src, Mat5 lReturn)
{
    int i, j, position;

    for (i = 0; i < 5; i++)
    {
        for (j = 0; j < 5; j++)
        {
            position = (row - 2 + i) * width + (column - 2 + j);
            lReturn[i][j] = *(src + position);
        }
    }
}/* End of function Populate_PixelMat5 */

void Populate_DenseWeights(float* src, float* dst, int size)
{
    int i = 0;
    do
    {
        *(dst + i) = *(src + i);
        i++;
    } while (i < size);
}

float Perform_Mat5Conv2D(Mat5 pixel, Mat5 kernel)
{
    int i, j;
    float lReturn = 0;

    for (i = 0; i < 5; i++)
    {
        for (j = 0; j < 5; j++)
        {
            lReturn += (float)kernel[i][j] * pixel[i][j];
        }
    }

    return lReturn;
}/* End of function Perform_Mat5Conv_1D */

void Conv2DLayer_KernelSize5(float* src, int height, int width, float* dst, Mat5 kernel)
{
    int x, y;
    Mat5 mat_pixel;

    for (x = 2; x < height - 2; x++)
    {
        for (y = 2; y < width - 2; y++)
        {
            Populate_PixelMat5(x, y, width, src, mat_pixel);
            float ret = Perform_Mat5Conv2D(mat_pixel, kernel);
            *(dst + (x - 2) * (width - 4) + (y - 2)) = ret;
        }
    }
}/* End of function Conv2DLayer_KernelSize5 */

void Populate_MatMaxPool5X2(int row, int column, int width, float* src, MatMaxPool5X2 lReturn)
{
    int i, position;

    for (i = 0; i < 5; i++)
    {
        position = (row + i) * width + (column);
        lReturn[i][0] = *(src + position);
        lReturn[i][0] = *(src + position + 1);
    }
}/* End of function Populate_PixelMat5 */

float Get_Maximum(MatMaxPool5X2 mat_pixel)
{
    float maximum = 0, temp = 0;
    int x, y;

    for (x = 0; x < 5; x++)
    {
        for (y = 0; y < 2; y++)
        {
            if (mat_pixel[x][y] > maximum)
                maximum = mat_pixel[x][y];
        }
    }

    return maximum;
}

void MaxPoolLayer(float* src, float* dst, int stride_width, int stride_height, int in_height, int in_width, int dst_width)
{
    int x, y;
    int i, j;
    MatMaxPool5X2 mat_pixel;

    int shape_height = in_height - in_height % stride_height;
    int shape_width = in_width - in_width % stride_width;

    for (x = 0, i = 0; x < shape_height; x += stride_height, i++)
    {
        for (y = 0, j = 0; y < shape_width; y += stride_width, j++)
        {
            Populate_MatMaxPool5X2(x, y, in_width, src, mat_pixel);
            *(dst + i * dst_width + j) = Get_Maximum(mat_pixel);
        }
    }
}/* End of function MaxPoolLayer */

float FullyConnectedLayer(float* src_pixel, float* kernel, int length)
{
    int i;
    float lReturn = 0;
    for (i = 0; i < length; i++)
        lReturn += (float)kernel[i] * src_pixel[i];

    return lReturn;
}/* End of function FullyConnectedLayer */

