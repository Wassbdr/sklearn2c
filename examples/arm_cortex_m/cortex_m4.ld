/* Minimal memory map of an STM32F407-class part: 1 MiB flash, 128 KiB SRAM. */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}
_estack = ORIGIN(RAM) + LENGTH(RAM);
ENTRY(Reset_Handler)
SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text   : { *(.text*) *(.rodata*) } > FLASH
  .data   : { *(.data*) } > RAM AT > FLASH
  .bss    : { *(.bss*) *(COMMON) } > RAM
  /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
