### Three/four layered neural networks for MNIST digit classification

The purpose is to compare coding the outputs as one-hot or binary. 

To understand all the possible flags you can set run 

``
th train-on-minst2.lua -h
``

In addition to varying number of units in the hidden layers (one or two), you
can also choose regularization. Setting -e="binary" will use the outputs as the
digits in binary (0 is treated as 10)
