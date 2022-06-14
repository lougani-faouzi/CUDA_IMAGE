#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "FreeImage.h"

#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

__global__ void Pop_art_Warho_stream(int * d_img, unsigned width, unsigned height, unsigned sub_image)
{

  int colonne = blockIdx.x * blockDim.x + threadIdx.x;
  int ligne = blockIdx.y * blockDim.y + threadIdx.y;

  int indx = ((ligne * width) + colonne) * 3;

  if ((ligne < height) && (colonne  < width))
  {
  
  // saturate block 2, with green
  if (sub_image == 2)
  {
    d_img[indx+1] = 255;
  }
  
  // saturate block 4, with blue
  if (sub_image == 4)
  {

    d_img[indx+2] = 255;
  } 
 
  // saturate block 3, with red
  if (sub_image == 3)
  {
    d_img[indx] = 255;
  }
  
  // leave block 1 unsaturated
  
  }

}
int main(int argc, char** argv)
{

  FreeImage_Initialise();
  const char *PathName = "img.jpg";
  const char *PathDest = "new_img.png";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if(! bitmap )
    exit( 1 ); //WTF?! We can't even allocate images ? Die !

  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap);

  fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

  int size = sizeof(unsigned int) * 3 * width * height;

  unsigned int *img = (unsigned int*) malloc(size);

  // convert image to array
  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }
  
  int *img1, *img2, *img3, *img4;
  int idx, indx_DROIT_HAUT, indx_DROIT_BAS, indx_GAUCHE_BAS;
  
  // allocate four sub-image host arrays with cudaMallocHost, because copy in asynchronous mode
  cudaMallocHost((void**)&img1, size/4 * sizeof(int));
  cudaMallocHost((void**)&img2, size/4 * sizeof(int));
  cudaMallocHost((void**)&img3, size/4 * sizeof(int));
  cudaMallocHost((void**)&img4, size/4 * sizeof(int));
  
  // for each sub-image, copy values from the full-size image 
  for ( int y =0; y<height/2; y++){
    for ( int x =0; x<width/2; x++){
      
      idx = ((y * width) + x) * 3;
      
      // compute array indices translations
      indx_DROIT_HAUT = ((y * width) + x + width/2) * 3;
      indx_DROIT_BAS = (((y+height/2)* width) + x + width/2) * 3;
      indx_GAUCHE_BAS = (((y+height/2) * width) + x) * 3;
      
      img1[idx + 0] = img[idx + 0];
      img1[idx + 1] = img[idx + 1];
      img1[idx + 2] = img[idx + 2];
      
      img2[idx + 0] = img[indx_DROIT_HAUT + 0];
      img2[idx + 1] = img[indx_DROIT_HAUT + 1];
      img2[idx + 2] = img[indx_DROIT_HAUT + 2];
      
      img3[idx + 0] = img[indx_GAUCHE_BAS + 0];
      img3[idx + 1] = img[indx_GAUCHE_BAS + 1];
      img3[idx + 2] = img[indx_GAUCHE_BAS + 2];
      
      img4[idx + 0] = img[indx_DROIT_BAS + 0];
      img4[idx + 1] = img[indx_DROIT_BAS + 1];
      img4[idx + 2] = img[indx_DROIT_BAS + 2];
    }
  }
  
  int *d_img1, *d_img2, *d_img3, *d_img4;
  
  // allocate four sub-image device arrays
  cudaMalloc((void**)&d_img1, size/4 * sizeof(int));
  cudaMalloc((void**)&d_img2, size/4 * sizeof(int));
  cudaMalloc((void**)&d_img3, size/4 * sizeof(int));
  cudaMalloc((void**)&d_img4, size/4 * sizeof(int));
  
  // create  four streams, one for each sub-image
  cudaStream_t stream[4];

  for (int i = 0; i < 4; ++i)
    cudaStreamCreate(&stream[i]);
  
  // copy data asynchronously from host to device
  cudaMemcpyAsync(d_img1, img1, size/4 * sizeof(int), cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(d_img2, img2, size/4 * sizeof(int), cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyAsync(d_img3, img3, size/4 * sizeof(int), cudaMemcpyHostToDevice, stream[2]);
  cudaMemcpyAsync(d_img4, img4, size/4 * sizeof(int), cudaMemcpyHostToDevice, stream[3]);
  
  // synchronize device with host
  cudaThreadSynchronize();

  dim3 dimGrid (128,72,1);
  dim3 dimBlock (30,30,1);
  
  // call 4 times the kernel, to saturate the image block, annotated  2  4 , each with the corresponding stream
  //                                                                  1  3
  Pop_art_Warho_stream<<<dimGrid, dimBlock, 0, stream[0]>>>(d_img1, width, height, 1);
  Pop_art_Warho_stream<<<dimGrid, dimBlock, 0, stream[1]>>>(d_img2, width, height, 2);
  Pop_art_Warho_stream<<<dimGrid, dimBlock, 0, stream[2]>>>(d_img3, width, height, 3);
  Pop_art_Warho_stream<<<dimGrid, dimBlock, 0, stream[3]>>>(d_img4, width, height, 4);
  
  // copy data asynchronisly from device to host
  cudaMemcpyAsync(img1, d_img1, size/4 * sizeof(int), cudaMemcpyDeviceToHost, stream[0]);
  cudaMemcpyAsync(img2, d_img2, size/4 * sizeof(int), cudaMemcpyDeviceToHost, stream[1]);
  cudaMemcpyAsync(img3, d_img3, size/4 * sizeof(int), cudaMemcpyDeviceToHost, stream[2]);
  cudaMemcpyAsync(img4, d_img4, size/4 * sizeof(int), cudaMemcpyDeviceToHost, stream[3]);
  
  // synchronize device with host
  cudaThreadSynchronize();
  
  // copy satured sub-images into the full-size image
  for ( int y =0; y<height/2; y++){
    for ( int x =0; x<width/2; x++){
      
      idx = ((y * width) + x) * 3;
      indx_DROIT_HAUT = ((y * width) + x + width/2) * 3;
      indx_DROIT_BAS = (((y+height/2)* width) + x + width/2) * 3;
      indx_GAUCHE_BAS = (((y+height/2) * width) + x) * 3;
      
      img[idx + 0] = img1[idx + 0];
      img[idx + 1] = img1[idx + 1];
      img[idx + 2] = img1[idx + 2];
      
      img[indx_DROIT_HAUT + 0] = img2[idx + 0];
      img[indx_DROIT_HAUT + 1] = img2[idx + 1];
      img[indx_DROIT_HAUT + 2] = img2[idx + 2];
      
      img[indx_GAUCHE_BAS + 0] = img3[idx + 0];
      img[indx_GAUCHE_BAS + 1] = img3[idx + 1];
      img[indx_GAUCHE_BAS + 2] = img3[idx + 2];
      
      img[indx_DROIT_BAS + 0] = img4[idx + 0];
      img[indx_DROIT_BAS + 1] = img4[idx + 1];
      img[indx_DROIT_BAS + 2] = img4[idx + 2];
    }
  }
  
  // save image into disk
  bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }

      pixel+=3;
    }
    // next line
    bits += pitch;
  }

  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    printf("Image successfully saved !\n");
  FreeImage_DeInitialise(); //Cleanup !
  
  // free memory from host and device
  
  cudaFree(d_img1);
  cudaFree(d_img2);
  cudaFree(d_img3);
  cudaFree(d_img4);

  cudaFreeHost(img1);
  cudaFreeHost(img2);
  cudaFreeHost(img3);
  cudaFreeHost(img4);
  
  free(img);

  return 0;
}
