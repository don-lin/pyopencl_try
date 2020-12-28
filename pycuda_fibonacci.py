import pycuda.autoinit
import pycuda.driver as drv
import numpy,time,math,random

from pycuda.compiler import SourceModule

def factor(n):
  s=1
  while(n>0):
    s*=n
    n-=1
  return s

mod = SourceModule("""
#include <cstdio>
__device__ float factor(float n){
  float result=1;
  while(n>0){
    result*=n;
    n--;
  }
  return result;
}

__device__ float fab(float n){
    float temp,b=1,r=0,i=0;

    while(i<n){
        temp=r;
        r+=b;
        b=temp;
        i++;
    }
    return r;
}

__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i]=fab(a[i]);
  dest[i]=sin(a[i]);
}
""")

multiply_them = mod.get_function("multiply_them")

a = numpy.random.randn(1000).astype(numpy.float32)
b = numpy.random.randn(1000).astype(numpy.float32)

for i in range(len(a)):
  a[i]=i;

dest = numpy.zeros_like(a)

start=time.time()

for i in range(10000):
  multiply_them(
          drv.Out(dest), drv.In(a), drv.In(b),
          block=(1000,1,1), grid=(1,1))


#print(dest-a*b/2)

print(dest)
print('time is:',time.time()-start)

start=time.time()

for j in range(10):
  for i in range(1000):
    factor(a[i])

print('time is:',time.time()-start)
