#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Read K=3 expert blocks per layer from the slab file using FILE_FLAG_NO_BUFFERING
// Measure: MB/s, ms per expert, ms per layer

#define SLAB_PATH "D:/flash-moe-engine/experts.slab"
#define SLOT_SIZE (3735552)       // 57 * 64KB, from expert_index.json
#define ALIGNMENT (65536)         // 64 KiB
#define NUM_LAYERS 60
#define EXPERTS_PER_LAYER 512
#define K 3

int main() {
    HANDLE hFile = CreateFileA(SLAB_PATH, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { printf("Failed to open slab\n"); return 1; }

    // Aligned buffer for unbuffered reads
    size_t buf_size = ((SLOT_SIZE + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    void* buf = _aligned_malloc(buf_size, ALIGNMENT);

    // Simulate 10 tokens: for each token, read K=3 random experts per layer
    int num_tokens = 10;
    LARGE_INTEGER total_bytes = {0};
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int tok = 0; tok < num_tokens; tok++) {
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            for (int k = 0; k < K; k++) {
                int expert_id = rand() % EXPERTS_PER_LAYER;
                long long offset = (long long)(layer * EXPERTS_PER_LAYER + expert_id) * SLOT_SIZE;

                LARGE_INTEGER li;
                li.QuadPart = (offset / ALIGNMENT) * ALIGNMENT;  // align down
                SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);

                DWORD bytesRead;
                ReadFile(hFile, buf, (DWORD)buf_size, &bytesRead, NULL);
                total_bytes.QuadPart += SLOT_SIZE;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mb = total_bytes.QuadPart / (1024.0 * 1024.0);

    printf("Tokens: %d\n", num_tokens);
    printf("Total reads: %d\n", num_tokens * NUM_LAYERS * K);
    printf("Total bytes: %.1f MB\n", mb);
    printf("Total time: %.1f ms\n", ms);
    printf("Bandwidth: %.1f MB/s\n", mb / (ms / 1000.0));
    printf("Per expert: %.2f ms\n", ms / (num_tokens * NUM_LAYERS * K));
    printf("Per layer (K=%d): %.2f ms\n", K, ms / (num_tokens * NUM_LAYERS));
    printf("Per token: %.1f ms (%.2f tok/s projected)\n", ms / num_tokens, num_tokens / (ms / 1000.0));

    _aligned_free(buf);
    CloseHandle(hFile);
    return 0;
}
