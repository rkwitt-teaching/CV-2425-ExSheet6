# Exercise set 6

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

### Exercise 6.1


**Step 1**

In this exercise, we will take images as inputs to a neural network in sort
of a special way. In particular, we will split our input images into **non-overlapping patches** of size $P \times P$ and linearly map each patch into a $D$-dimensional representation (i.e., a vector) via (1) an appropriately
configured 2D convolution operation (without bias) and (2) rearranging the output of that operation. *We can safely assume that the user will set the patch size such that it perfectly fits into the width and height of the image.*

For example, if our input images are of size `(3,32,32)` and we want $4 \times 4$ patches (i.e., $P=4$) and an embedding dimension of $D=128$, we should get as an output a tensor of size `(64,128)`, as we have 64 non-overlapping $4 \times 4$ patches. We will denote the number of non-overlapping patches as $N$ from here on. As this should work for *batches* of images, e.g., `(16,3,32.32)`, the output should really be `(16,64,128)` in this example.

Please implement this functionality within the provided `Encoder` class, which takes three arguments:

```python
...
def __init__(self, 
    in_channels: int = 3,   # nr. of input channels of the image
    patch_size: int = 16,   # our patch size (P)
    emb_size: int = 768):   # our embedding size (D)

    self.conv_layer = ...
...                 
```

**Please do not modify the variable names!**

**Step 2**

After **Step 1**, we have (in our example) an output of size `(64,128)`. In case we have 10 images, we would have an output of size `(10,64,128)`. In **Step 2**, we will concatenate to the $N$ $D$-dimensional embeddings of each image, a $D$-dimensional vector of all **ones**, so that the output tensor is actually of size `(65,128)` or `(10,65,128)` in the example(s) from above. This vector should be a `torch.nn.Parameter` and, in particular, we want this concatenation to work for any input size (so not just $32 \times 32$ images). Hence, we need a variable (we call it `self.token` in the template code) within the `Encoder` class that is instantiated with the correct size and later (within the `forward` method) *repeated* to match the output size after **Step 1** (see Implementation notes below!).

### Implementation notes

Use `einops` for rearranging the output of the 2D convolution (specifically the function `rearrange`); this is specifially relevant for **Step 1**; also, in case you need to repeat a vector along some dimension, use the `repeat` function (also from the `einops` library); this is specificially relevant for **Step 2**.

The `forward` method within the `Encoder` class should return ***two** tensors: (1) the tensor for **Step 1** and the tensor after **Step 2**. In case you decide **not** to implement **Step 2**, just return `None`, e.g.,

```python
def forward(self, x):
    ...
    return output0, None
```

Obviously, this will cause the test for **Step 2** to fail!