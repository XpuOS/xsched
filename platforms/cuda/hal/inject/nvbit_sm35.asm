	code for sm_35
		Function : instrumented_kernel
                EntryPoint: 0xb1740
	.headerflags    @"EF_CUDA_SM35 EF_CUDA_PTX_SM(EF_CUDA_SM35)"
                                                                                /* 0x0880b8b0a0a08cac */
        /*0008*/                   JMP 0xb1cf8;                                 /* 0x1080058e7c1c003c */
        /*0010*/                   S2R R0, SR_CTAID.X;                          /* 0x86400000129c0002 */
        /*0018*/                   S2R R3, SR_TID.X;                            /* 0x86400000109c000e */
        /*0020*/                   IMAD R0, R0, c[0x0][0x28], R3;               /* 0x51080c00051c0002 */
        /*0028*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x158], PT;  /* 0x5b681c002b1c001e */
        /*0030*/               @P0 EXIT;                                        /* 0x180000000000003c */
        /*0038*/                   ISCADD R4.CC, R0, c[0x0][0x140], 0x3;        /* 0x60c40c00281c0012 */
                                                                                /* 0x08b010a0b010a0ac */
        /*0048*/                   MOV32I R9, 0x8;                              /* 0x74000000041fc026 */
        /*0050*/                   IMAD.HI.X R5, R0, R9, c[0x0][0x144];         /* 0x93182400289c0016 */
        /*0058*/                   ISCADD R6.CC, R0, c[0x0][0x148], 0x3;        /* 0x60c40c00291c001a */
        /*0060*/                   LD.E.CG.64 R4, [R4];                         /* 0xcd800000001c1010 */
        /*0068*/                   IMAD.HI.X R7, R0, R9, c[0x0][0x14c];         /* 0x93182400299c001e */
        /*0070*/                   LD.E.CG.64 R2, [R6];                         /* 0xcd800000001c1808 */
        /*0078*/                   ISCADD R8.CC, R0, c[0x0][0x150], 0x3;        /* 0x60c40c002a1c0022 */
                                                                                /* 0x08000000b810a4fc */
        /*0088*/                   IMAD.HI.X R9, R0, R9, c[0x0][0x154];         /* 0x931824002a9c0026 */
        /*0090*/                   DADD R2, R2, R4;                             /* 0xe3800000021c080a */
        /*0098*/                   ST.E.64 [R8], R2;                            /* 0xe5800000001c2008 */
        /*00a0*/                   EXIT;                                        /* 0x18000000001c003c */
        /*00a8*/                   BRA 0xa8;                                    /* 0x12007ffffc1c003c */
        /*00b0*/                   NOP;                                         /* 0x85800000001c3c02 */
        /*00b8*/                   NOP;                                         /* 0x85800000001c3c02 */



	code for sm_35
		Function : instrument_code
                EntryPoint: 0xb1cf8
	.headerflags    @"EF_CUDA_SM35 EF_CUDA_PTX_SM(EF_CUDA_SM35)"
        /*0000*/                   JCAL 0xb1c00;             /* 0x1100058e00000100 */
        /*0008*/                   LDC R1, c[0x0][0x44];     /* 0x7ca00000221ffc06 */
        /*0010*/                   LDC R4, c[0x0][0x1880];   /* 0x7ca0000c401ffc12 */
        /*0018*/                   LDC R5, c[0x0][0x1884];   /* 0x7ca0000c421ffc16 */
        /*0020*/                   LDC R6, c[0x0][0x1888];   /* 0x7ca0000c441ffc1a */
        /*0028*/                   LDC R7, c[0x0][0x188c];   /* 0x7ca0000c461ffc1e */
        /*0030*/                   LDC R8, c[0x0][0x1890];   /* 0x7ca0000c481ffc22 */
        /*0038*/                   LDC R9, c[0x0][0x1894];   /* 0x7ca0000c4a1ffc26 */
        /*0040*/                   LDC R10, c[0x0][0x1898];  /* 0x7ca0000c4c1ffc2a */
        /*0048*/                   LDC R11, c[0x0][0x189c];  /* 0x7ca0000c4e1ffc2e */
        /*0050*/                   LDC R12, c[0x0][0x18a0];  /* 0x7ca0000c501ffc32 */
        /*0058*/                   LDC R13, c[0x0][0x18a4];  /* 0x7ca0000c521ffc36 */
        /*0060*/                   LDC R14, c[0x0][0x18a8];  /* 0x7ca0000c541ffc3a */
        /*0068*/                   LDC R15, c[0x0][0x18ac];  /* 0x7ca0000c561ffc3e */
        /*0070*/                   JCAL 0xadac0;             /* 0x1100056d60000100 */
        /*0078*/                   JCAL 0xb1c80;             /* 0x1100058e40000100 */
        /*0080*/                   MOV R1, c[0x0][0x44];     /* 0x64c03c00089c0006 */
        /*0088*/                   JMP 0xb1750;              /* 0x1080058ba81c003c */
        /*0090*/                   NOP;                      /* 0x85800000001c3c02 */
        /*0098*/                   NOP;                      /* 0x85800000001c3c02 */
        /*00a0*/                   NOP;                      /* 0x85800000001c3c02 */



	code for sm_35
		Function : nvbit_functions
                EntryPoint: 0xb1c00
	.headerflags    @"EF_CUDA_SM35 EF_CUDA_PTX_SM(EF_CUDA_SM35)"
        /*0000*/                   NOP;                     /* 0x85800000001c3c02 */
        /*0008*/                   NOP;                     /* 0x85800000001c3c02 */
        /*0010*/                   TEXDEPBAR 0x0;           /* 0x77000000001c0002 */
        /*0018*/                   STL [0xfffe00], R0;      /* 0x7aa07fff001ffc02 */
        /*0020*/                   STL [0xfffe04], R1;      /* 0x7aa07fff021ffc06 */
        /*0028*/                   STL [0xfffe08], R2;      /* 0x7aa07fff041ffc0a */
        /*0030*/                   STL [0xfffe0c], R3;      /* 0x7aa07fff061ffc0e */
        /*0038*/                   STL [0xfffe10], R4;      /* 0x7aa07fff081ffc12 */
        /*0040*/                   STL [0xfffe14], R5;      /* 0x7aa07fff0a1ffc16 */
        /*0048*/                   STL [0xfffe18], R6;      /* 0x7aa07fff0c1ffc1a */
        /*0050*/                   STL [0xfffe1c], R7;      /* 0x7aa07fff0e1ffc1e */
        /*0058*/                   STL [0xfffe20], R8;      /* 0x7aa07fff101ffc22 */
        /*0060*/                   STL [0xfffe24], R9;      /* 0x7aa07fff121ffc26 */
        /*0068*/                   P2R R4, PR, RZ, 0xffff;  /* 0xc640007fff9ffc11 */
        /*0070*/                   STL [0xfffe60], R4;      /* 0x7aa07fff301ffc12 */
        /*0078*/                   RET;                     /* 0x19000000001c003c */

                PC: 0xb1c80
        /*0080*/                   NOP;                     /* 0x85800000001c3c02 */
        /*0088*/                   NOP;                     /* 0x85800000001c3c02 */
        /*0090*/                   LDL R4, [0xfffe60];      /* 0x7a207fff301ffc12 */
        /*0098*/                   R2P PR, R4;              /* 0xc680007fff9c1001 */
        /*00a0*/                   LDL R0, [0xfffe00];      /* 0x7a207fff001ffc02 */
        /*00a8*/                   LDL R1, [0xfffe04];      /* 0x7a207fff021ffc06 */
        /*00b0*/                   LDL R2, [0xfffe08];      /* 0x7a207fff041ffc0a */
        /*00b8*/                   LDL R3, [0xfffe0c];      /* 0x7a207fff061ffc0e */
        /*00c0*/                   LDL R4, [0xfffe10];      /* 0x7a207fff081ffc12 */
        /*00c8*/                   LDL R5, [0xfffe14];      /* 0x7a207fff0a1ffc16 */
        /*00d0*/                   LDL R6, [0xfffe18];      /* 0x7a207fff0c1ffc1a */
        /*00d8*/                   LDL R7, [0xfffe1c];      /* 0x7a207fff0e1ffc1e */
        /*00e0*/                   LDL R8, [0xfffe20];      /* 0x7a207fff101ffc22 */
        /*00e8*/                   LDL R9, [0xfffe24];      /* 0x7a207fff121ffc26 */
        /*00f0*/                   RET;                     /* 0x19000000001c003c */

                PC: 0xb1cf8
        /*00f8*/                   JCAL 0xb1c00;            /* 0x1100058e00000100 */
        /*0100*/                   LDC R1, c[0x0][0x44];    /* 0x7ca00000221ffc06 */
        /*0108*/                   MOV32I R4, 0x1;          /* 0x74000000009fc012 */
        /*0110*/                   MOV32I R5, 0x0;          /* 0x74000000001fc016 */
        /*0118*/                   LDC R6, c[0x0][0x1880];  /* 0x7ca0000c401ffc1a */
        /*0120*/                   LDC R7, c[0x0][0x1884];  /* 0x7ca0000c421ffc1e */
        /*0128*/                   JCAL 0xadac0;            /* 0x1100056d60000100 */
        /*0130*/                   JCAL 0xb1c80;            /* 0x1100058e40000100 */
        /*0138*/                   MOV R1, c[0x0][0x44];    /* 0x64c03c00089c0006 */
        /*0140*/                   JMP 0xb1750;             /* 0x1080058ba81c003c */
        /*0148*/                   NOP;                     /* 0x85800000001c3c02 */
        /*0150*/                   NOP;                     /* 0x85800000001c3c02 */
        /*0158*/                   NOP;                     /* 0x85800000001c3c02 */



	code for sm_35
		Function : injected_function
                EntryPoint: 0xadac0
	.headerflags    @"EF_CUDA_SM35 EF_CUDA_PTX_SM(EF_CUDA_SM35)"
                                                                   /* 0x08b8b8bcb8a0a010 */
        /*0008*/                   MOV R3, R1;                     /* 0xe4c03c00009c000e */
        /*0010*/                   IADD32I R1, R1, -0x40;          /* 0x407fffffe01c0405 */
        /*0018*/                   LOP32I.AND R1, R1, 0xfffffff0;  /* 0x207ffffff81c0404 */
        /*0020*/                   STL [R1+0x30], R2;              /* 0x7aa00000181c040a */
        /*0028*/                   MOV R2, R3;                     /* 0xe4c03c00019c000a */
        /*0030*/                   STL.128 [R1+0x10], R8;          /* 0x7ab00000081c0422 */
        /*0038*/                   STL.128 [R1+0x20], R12;         /* 0x7ab00000101c0432 */
                                                                   /* 0x0880bcb8808010e8 */
        /*0048*/                   STL.128 [R1], R4;               /* 0x7ab00000001c0412 */
        /*0050*/                   LOP.OR R6, R1, c[0x0][0x24];    /* 0x62001000049c041a */
        /*0058*/                   MOV R7, RZ;                     /* 0xe4c03c007f9c001e */
        /*0060*/                   MOV32I R4, 0x4060000;           /* 0x74020300001fc012 */
        /*0068*/                   MOV32I R5, 0x42;                /* 0x74000000211fc016 */
        /*0070*/                   JCAL 0x821c0;                   /* 0x11000410e01c0100 */
        /*0078*/                   MOV R3, R2;                     /* 0xe4c03c00011c000e */
                                                                   /* 0x0800000000bc10b8 */
        /*0088*/                   LDL R2, [R1+0x30];              /* 0x7a200000181c040a */
        /*0090*/                   MOV R1, R3;                     /* 0xe4c03c00019c0006 */
        /*0098*/                   RET;                            /* 0x19000000001c003c */
        /*00a0*/                   BRA 0xa0;                       /* 0x12007ffffc1c003c */
        /*00a8*/                   NOP;                            /* 0x85800000001c3c02 */
        /*00b0*/                   NOP;                            /* 0x85800000001c3c02 */
        /*00b8*/                   NOP;                            /* 0x85800000001c3c02 */
