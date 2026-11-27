/* Bare-metal Cortex-M4 firmware skeleton that links a sklearn2c model.
 * It only needs to compile and link: the point is the flash/RAM report. */
#include <stdint.h>

int32_t model_predict(const float *x); /* from the generated model.c */

volatile float sensor[16];   /* written by an ISR / DMA in a real firmware */
volatile int32_t prediction; /* read by the application */

void Reset_Handler(void) {
    for (;;) prediction = model_predict((const float *)sensor);
}

void Default_Handler(void) { for (;;) {} }

extern uint32_t _estack;
__attribute__((section(".isr_vector"), used))
const void *vectors[16] = {
    &_estack, Reset_Handler, Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, 0, 0, 0, 0, Default_Handler, Default_Handler, 0, Default_Handler, Default_Handler,
};
