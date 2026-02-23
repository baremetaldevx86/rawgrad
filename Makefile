
CC      = gcc
CFLAGS  = -Wall -g -O2 -Iinclude
LDFLAGS = -lm
VPATH   = src:examples:tests

all: train train_xor train_mnist

train: train.o engine.o nn.o mlp.o loss.o optim.o
	$(CC) $(CFLAGS) -o train train.o engine.o nn.o mlp.o loss.o optim.o $(LDFLAGS)

train_xor: train_xor.o engine.o nn.o mlp.o loss.o optim.o
	$(CC) $(CFLAGS) -o train_xor train_xor.o engine.o nn.o mlp.o loss.o optim.o $(LDFLAGS)

train_mnist: train_mnist.o engine.o nn.o mlp.o loss.o optim.o mnist_loader.o
	$(CC) $(CFLAGS) -o train_mnist train_mnist.o engine.o nn.o mlp.o loss.o optim.o mnist_loader.o $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o train train_xor train_mnist
