import numpy as np
import pyopencl as cl

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)

for i in range(len(a_np)):
    a_np[i]=i

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
float factor(float a){
    float result=1;
    while(a>0){
        result*=a;
        a-=1;
    }
    return result;
}
float isPrime(float num){
    int value=num;
    for(int i=2;i<num;i++){
        if(value%i==0)
            return 0;
    }
    return 1;
}

__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
    int gid = get_global_id(0);
    res_g[gid] = a_g[gid] * b_g[gid];
    res_g[gid]=factor(a_g[gid]);
    res_g[gid]=isPrime(a_g[gid]);
}
""").build()

def isPrime(n):
    for i in range(2,n):
        if n%i==0:
            return False
    return True

import time
start=time.time()
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
for i in range(1):
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)


# Check on CPU with Numpy:
for i in range(len(res_np)):
    if res_np[i]==1:
        continue
        #print(i,end=' ')

print(time.time()-start)
start=time.time()

for i in range(50000):
    if isPrime(i):
        continue
        #print(i,end=' ')
print(time.time()-start)
