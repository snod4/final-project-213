CC := nvcc
TARGETS := make-hash-file local-crack

all: $(TARGETS)

clean:
	rm -rf $(TARGETS)

%: %.cu
	@$(CC) -o $@ $<
