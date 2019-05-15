CC := nvcc
TARGETS := make-hash-file local-crack distributed-crack crackDrone

all: $(TARGETS)

clean:
	rm -rf $(TARGETS)

%: %.cu
	@$(CC) -o $@ $<
