#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdint.h>
#include <math.h>
#include <string.h>

#define WIDTH                           (44U)
#define HEIGHT                          (73U)
#define FRAME_SIZE                      (WIDTH * HEIGHT)

#define MODEL_WEIGHTS                   (253U)
#define KERNEL_WEIGHTS                  (25U)
#define DENSE_WEIGHTS                   (120U)

#define L2_WIDTH                        (40U)
#define L2_HEIGHT                       (69U)

#define L3_WIDTH                        (36U)
#define L3_HEIGHT                       (65U)

#define L4_WIDTH                        (32U)
#define L4_HEIGHT                       (61U)

#define L5_WIDTH                        (28U)
#define L5_HEIGHT                       (57U)

#define MAXPOOL_WIDTH                   (24U)
#define MAXPOOL_HEIGHT                  (53U)

#define DENSE_WIDTH                     (12U)
#define DENSE_HEIGHT                    (10U)


#include <stdio.h>
#include <stdlib.h>

/** Macro to indicate function returned with expected operation */
#define E_OK        (0u)

/** Macro to indicate function returned with unexpected operation */
#define E_NOT_OK    (1u)

/** Type definition for standard return type values to integer */
typedef int Std_ReturnType;

/** 5X5 Matrix to store pixel data and kernel matrices */
typedef float Mat5[5][5];

/** 5X5 Matrix to store pixel data and kernel matrices */
typedef float MatMaxPool5X2[5][2];

/** Creating Global Place holders for Weights*/
Mat5 Kernel_L1, Kernel_L2, Kernel_L3, Kernel_L4, Kernel_L5;
float Bais_L1, Bais_L2, Bais_L3, Bais_L4, Bais_L5, Bais_L7, Bais_L8;
float* Weights_L7 = (float*)malloc(DENSE_WEIGHTS * sizeof(float));
float Weight_L8;

void Populate_KernelMat5(float* src, Mat5 lReturn);
void Populate_DenseWeights(float* src, float* dst, int size);
void Populate_PixelMat5(int row, int column, int width, float* src, Mat5 lReturn);
void Conv2DLayer_KernelSize5(float* src, int height, int width, float* dst, Mat5 kernel);
void AddBais(float val_bais, int src_size, float* src);
float Get_Maximum(MatMaxPool5X2 mat_pixel);
void Perform_MaxPool(int row, int column, float* src, int in_width, float* dst);
void MaxPoolLayer(float* src, float* dst, int stride_width, int stride_height, int in_height, int in_width, int dst_width);
float FullyConnectedLayer(float* src_pixel, float* kernel, int length);
float Sigmoid(float value);
void Read_ModelParamaters(char* fname);
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
    printf("\n\n");

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
    /*Layer 2*/
    float* audio_out_layer2 = (float*)malloc(L3_WIDTH * L3_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer1, L2_HEIGHT, L2_WIDTH, audio_out_layer2, Kernel_L2);
    AddBais(Bais_L2, L3_WIDTH * L3_HEIGHT, audio_out_layer2);
    /*Layer 3*/
    float* audio_out_layer3 = (float*)malloc(L4_WIDTH * L4_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer2, L3_HEIGHT, L3_WIDTH, audio_out_layer3, Kernel_L3);
    AddBais(Bais_L3, L4_WIDTH * L4_HEIGHT, audio_out_layer3);
    /*Layer 4*/
    float* audio_out_layer4 = (float*)malloc(L5_WIDTH * L5_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer3, L4_HEIGHT, L4_WIDTH, audio_out_layer4, Kernel_L4);
    AddBais(Bais_L4, L5_WIDTH * L5_HEIGHT, audio_out_layer4);
    /*Layer 5*/
    float* audio_out_layer5 = (float*)malloc(MAXPOOL_WIDTH * MAXPOOL_HEIGHT * sizeof(float));
    Conv2DLayer_KernelSize5(audio_out_layer4, L5_HEIGHT, L5_WIDTH, audio_out_layer5, Kernel_L5);
    AddBais(Bais_L5, MAXPOOL_WIDTH * MAXPOOL_HEIGHT, audio_out_layer5);
    /*Layer 6 - MaxPooling of size (5,2)*/
    float* audio_out_layer6 = (float*)malloc(DENSE_WIDTH * DENSE_HEIGHT * sizeof(float));
    MaxPoolLayer(audio_out_layer5, audio_out_layer6, 2, 5, MAXPOOL_HEIGHT, MAXPOOL_WIDTH, DENSE_WIDTH);

    float temp = FullyConnectedLayer(audio_out_layer6, Weights_L7, DENSE_WEIGHTS) + Bais_L7;
    if (temp < 0)
        temp = (float)exp(temp) - 1;

    float retval = temp * Weight_L8 + Bais_L8;
    retval = Sigmoid(retval);

    printf("Predicted Audio: %f \n", retval);
    return retval;
}

void AddBais(float val_bais, int size, float* src)
{
    int iter = 0;
    do
    {
        *(src + iter) += val_bais;
        if (*(src + iter) < 0)
            *(src + iter) = (float)exp(*(src + iter)) - 1;

        iter++;
    } while (iter < size);
}

float Sigmoid(float value)
{
    return (float)1 / (1 + exp(-value));
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
    for (i = 0; i < 120; i++)
        lReturn += (float)kernel[i] * src_pixel[i];

    return lReturn;
}/* End of function FullyConnectedLayer */

