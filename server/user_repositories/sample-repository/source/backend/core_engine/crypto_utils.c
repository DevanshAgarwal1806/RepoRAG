#include <string.h>

/**
 * Applies a basic bitwise XOR masking for lightweight payload obfuscation.
 * This is used for encrypting telemetry data before transit.
 */
void mask_payload(const char* input, char* output, char key) {
    int i = 0;
    while (input[i] != '\0') {
        output[i] = input[i] ^ key;
        i++;
    }
    output[i] = '\0';
}