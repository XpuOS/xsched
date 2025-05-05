
Fatbin elf code:
================
arch = sm_35
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

	code for sm_35
		Function : check_preempt
	.headerflags    @"EF_CUDA_SM35 EF_CUDA_PTX_SM(EF_CUDA_SM35)"

        /*0010*/                   TEXDEPBAR 0x0;                       /* 0x77000000001c0002 */
        /*0010*/                   LDC R4, c[0x0][0x1880];              /* 0x7ca0000c401ffc12 */
        /*0018*/                   LDC R5, c[0x0][0x1884];              /* 0x7ca0000c421ffc16 */
        /*0020*/                   LDC R6, c[0x0][0x1890];              /* 0x7ca0000c481ffc1a */
        /*0028*/                   LDC R7, c[0x0][0x1894];              /* 0x7ca0000c4a1ffc1e */
                                                                        /* 0x08a088808c8c8c10 */
        /*0008*/                   S2R R0, SR_CTAID.X;                  /* 0x86400000129c0002 */
        /*0010*/                   SSY 0x160;                           /* 0x14800000a4000000 */
        /*0018*/                   S2R R3, SR_CTAID.Y;                  /* 0x86400000131c000e */
        /*0020*/                   S2R R9, SR_CTAID.Z;                  /* 0x86400000139c0026 */
        /*0028*/                   S2R R8, SR_TID.X;                    /* 0x86400000109c0022 */
        /*0030*/                   IMAD R0, R0, c[0x0][0x38], R3;       /* 0x51080c00071c0002 */
        /*0038*/                   S2R R3, SR_TID.Y;                    /* 0x86400000111c000e */
                                                                        /* 0x08b0a010a09c8010 */
        /*0048*/                   S2R R10, SR_TID.Z;                   /* 0x86400000119c002a */
        /*0050*/                   LOP.OR R3, R8, R3;                   /* 0xe2001000019c200e */
        /*0058*/                   IMAD R0, R0, c[0x0][0x3c], R9;       /* 0x51082400079c0002 */
        /*0060*/                   LOP.OR R3, R3, R10;                  /* 0xe2001000051c0c0e */
        /*0068*/                   ISCADD R0, R0, 0x4, 0x1;             /* 0xc0c00400021c0001 */
        /*0070*/                   ISETP.NE.AND P0, PT, R3, RZ, PT;     /* 0xdb581c007f9c0c1e */
        /*0078*/                   IMAD.U32.U32 R8.CC, R0, 0x4, R4;     /* 0xa0041000021c0021 */
                                                                        /* 0x08a0bcb0fc10bc10 */
        /*0088*/                   IMAD.U32.U32.HI.X R9, R0, 0x4, R5;   /* 0xa2101400021c0025 */
        /*0090*/               @P0 NOP.S;                               /* 0x8580000000403c02 */
        /*0098*/                   LD.E.CG R0, [R4];                    /* 0xcc800000001c1000 */
        /*00a0*/                   PBK 0x150;                           /* 0x1500000054000000 */
        /*00a8*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;     /* 0xdb581c007f9c001e */
        /*00b0*/              @!P0 BRA 0x138;                           /* 0x120000004020003c */
        /*00b8*/                   MOV32I R3, 0x1;                      /* 0x74000000009fc00e */
                                                                        /* 0x08b0bcb0b0fcb810 */
        /*00c8*/                   ST.E [R8], R3;                       /* 0xe4800000001c200c */
        /*00d0*/                   SSY 0x128;                           /* 0x1480000028000000 */
        /*00d8*/                   LD.E.CG.64 R10, [R4+0x8];            /* 0xcd800000041c1028 */
        /*00e0*/                   IADD RZ.CC, -RZ, R10;                /* 0xe0940000051ffffe */
        /*00e8*/                   ISETP.NE.X.AND P0, PT, R11, RZ, PT;  /* 0xdb585c007f9c2c1e */
        /*00f0*/              @!P0 BRA 0x120;                           /* 0x120000001420003c */
        /*00f8*/                   IADD RZ.CC, -R6, R10;                /* 0xe0940000051c1bfe */
                                                                        /* 0x0880b810bcb8bcb0 */
        /*0108*/                   ISETP.NE.X.AND P0, PT, R11, R7, PT;  /* 0xdb585c00039c2c1e */
        /*0110*/              @!P0 NOP.S;                               /* 0x8580000000603c02 */
        /*0118*/                   BRK;                                 /* 0x1a000000001c003c */
        /*0120*/                   ST.E.64.S [R4+0x8], R6;              /* 0xe5800000045c1018 */
        /*0128*/                   ST.E [R8+0x4], R3;                   /* 0xe4800000021c200c */
        /*0130*/                   BRK;                                 /* 0x1a000000001c003c */
        /*0138*/                   ST.E [R8], RZ;                       /* 0xe4800000001c23fc */
                                                                        /* 0x08bcb0fc00bcbcbc */
        /*0148*/                   BRK;                                 /* 0x1a000000001c003c */
        /*0150*/                   NOP;                                 /* 0x85800000001c3c02 */
        /*0158*/                   NOP.S;                               /* 0x85800000005c3c02 */
        /*0160*/                   BAR.SYNC 0x0;                        /* 0x8540dc00001c0002 */
        /*0168*/                   LD.E.CG R8, [R8];                    /* 0xcc800000001c2020 */
        /*0170*/                   ISETP.NE.AND P0, PT, R8, RZ, PT;     /* 0xdb581c007f9c201e */
        /*0178*/              @!P0 BRA 0x198;                           /* 0x120000000c20003c */
                                                                        /* 0x0800000000bcbc10 */
        /*0188*/                   MOV RZ, RZ;                          /* 0xe4c03c007f9c03fe */
        /*0190*/                   EXIT;                                /* 0x18000000001c003c */
		..........


		Function : restore_exec
	.headerflags    @"EF_CUDA_SM35 EF_CUDA_PTX_SM(EF_CUDA_SM35)"
                                                                       /* 0x08a0a0908c8cbcbc */
        /*0008*/                   LDC R4, c[0x0][0x1880];             /* 0x7ca0000c401ffc12 */
        /*0010*/                   LDC R5, c[0x0][0x1884];             /* 0x7ca0000c421ffc16 */
        /*0018*/                   S2R R0, SR_CTAID.X;                 /* 0x86400000129c0002 */
        /*0020*/                   S2R R3, SR_CTAID.Y;                 /* 0x86400000131c000e */
        /*0028*/                   S2R R6, SR_CTAID.Z;                 /* 0x86400000139c001a */
        /*0030*/                   IMAD R0, R0, c[0x0][0x38], R3;      /* 0x51080c00071c0002 */
        /*0038*/                   IMAD R0, R0, c[0x0][0x3c], R6;      /* 0x51081800079c0002 */
                                                                       /* 0x08b8bcb0fca0b0a0 */
        /*0048*/                   ISCADD R0, R0, 0x5, 0x1;            /* 0xc0c00400029c0001 */
        /*0050*/                   IMAD.U32.U32 R4.CC, R0, 0x4, R4;    /* 0xa0041000021c0011 */
        /*0058*/                   IMAD.U32.U32.HI.X R5, R0, 0x4, R5;  /* 0xa2101400021c0015 */
        /*0060*/                   LD.E R0, [R4];                      /* 0xc4800000001c1000 */
        /*0068*/                   ISETP.NE.AND P0, PT, R0, RZ, PT;    /* 0xdb581c007f9c001e */
        /*0070*/              @!P0 EXIT;                               /* 0x180000000020003c */
        /*0078*/                   BAR.SYNC 0x0;                       /* 0x8540dc00001c0002 */
                                                                       /* 0x08000000bcbcbc10 */
        /*0088*/                   ST.E [R4], RZ;                      /* 0xe4800000001c13fc */
        /*0090*/                   LDC R0, c[0x0][0x1888];             /* 0x7ca0000c441ffc02 */
        /*0098*/                   LDC R1, c[0x0][0x188c];             /* 0x7ca0000c461ffc06 */
        /*0030*/                   JMX R0;                             /* 0x10000000001c003c */
        /*00a8*/                   BRA 0xa8;                           /* 0x12007ffffc1c003c */
        /*00b0*/                   NOP;                                /* 0x85800000001c3c02 */
        /*00b8*/                   NOP;                                /* 0x85800000001c3c02 */
		..........
