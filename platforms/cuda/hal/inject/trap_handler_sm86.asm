
	code for sm_86
		Function : trap_handler
                Entry Point : 0x7fac32f69a00
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"

// trap_entry_bootstrap begin
        /*0000*/                   BPT.DRAIN;                                     /* 0x000000000000795c */
                                                                                  /* 0x000fc00000500000 */
        /*0010*/                   BMOV.32 MACTIVE, 0xffffffff;                   /* 0xffffffff1a007956 */
                                                                                  /* 0x000fc00000000000 */
        /*0020*/                   STL.128 [0xffffd0], R4 ;                       /* 0xffffd004ff007387 */
                                                                                  /* 0x0009c00000100c00 */
        /*0030*/                   STL.128 [0xffffe0], R8 ;                       /* 0xffffe008ff007387 */
                                                                                  /* 0x0003c00000100c00 */
        /*0040*/                   STL.128 [0xffffc0], R0 ;                       /* 0xffffc000ff007387 */
                                                                                  /* 0x0005c00000100c00 */
        /*0050*/                   MOV R4, R12 ;                                  /* 0x0000000c00047202 */
                                                                                  /* 0x010fc00000000f00 */
        /*0060*/                   MOV R5, R13 ;                                  /* 0x0000000d00057202 */
                                                                                  /* 0x010fc00000000f00 */
        /*0070*/                   P2R R6, PR, RZ, 0xff ;                         /* 0x000000ffff067803 */
                                                                                  /* 0x010fc00000000000 */
        /*0080*/                   BMOV.32 R7, TRAP_RETURN_MASK ;                 /* 0x0000000017077355 */
                                                                                  /* 0x010f400000000000 */
        /*0090*/                   STL.128 [0xfffff0], R4 ;                       /* 0xfffff004ff007387 */
                                                                                  /* 0x0209c00000100c00 */
        /*00a0*/                   RPCMOV.32 R8, Rpc.LO ;                         /* 0x0000000000087353 */
                                                                                  /* 0x002fc00000000000 */
        /*00b0*/                   RPCMOV.32 R9, Rpc.HI ;                         /* 0x0000000080097353 */
                                                                                  /* 0x002fc00000000000 */
        /*00c0*/                   BMOV.32 R10, OPT_STACK ;                       /* 0x000000001c0a7355 */
                                                                                  /* 0x002f000000000000 */
        /*00d0*/                   S2R R4, SR_SM_SPA_VERSION ;                    /* 0x0000000000047919 */
                                                                                  /* 0x010f000000002c00 */
        /*00e0*/                   ISETP.NE.U32.AND P4, PT, RZ, UR2, PT;          /* 0x00000002ff007c0c */
                                                                                  /* 0x000fc0000bf85070 */
        /*00f0*/                   ISETP.NE.U32.OR P4, PT, R4, -0x7f87fa00, P4 ;  /* 0x807806000400780c */
                                                                                  /* 0x010fc00002785470 */
        /*0100*/               @P4 BMOV.32.CLEAR R11, B0 ;                        /* 0x00000000000b4355 */
                                                                                  /* 0x002f000000100000 */
        /*0110*/               @P4 BRA.U 0x250;                                   /* 0x0000013100004947 */
                                                                                  /* 0x000fc00003800000 */
        /*0120*/                   MOV R11, 0x0 ;                                 /* 0x00000000000b7802 */
                                                                                  /* 0x012fe20000000f00 */
        /*0130*/                   BMOV.32 B0, 0xffffffff ;                       /* 0xffffffff00007956 */
                                                                                  /* 0x000fe20000000000 */
        /*0140*/                   BMOV.32.CLEAR B1, B0 ;                         /* 0x0000000000017f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0150*/                   BMOV.32.CLEAR B2, B1 ;                         /* 0x0000000001027f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0160*/                   BMOV.32.CLEAR B3, B2 ;                         /* 0x0000000002037f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0170*/                   BMOV.32.CLEAR B4, B3 ;                         /* 0x0000000003047f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0180*/                   BMOV.32.CLEAR B5, B4 ;                         /* 0x0000000004057f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0190*/                   BMOV.32.CLEAR B6, B5 ;                         /* 0x0000000005067f55 */
                                                                                  /* 0x000fe20000100000 */
        /*01a0*/                   BMOV.32.CLEAR B7, B6 ;                         /* 0x0000000006077f55 */
                                                                                  /* 0x000fe20000100000 */
        /*01b0*/                   BMOV.32.CLEAR B8, B7 ;                         /* 0x0000000007087f55 */
                                                                                  /* 0x000fe20000100000 */
        /*01c0*/                   BMOV.32.CLEAR B9, B8 ;                         /* 0x0000000008097f55 */
                                                                                  /* 0x000fe20000100000 */
        /*01d0*/                   BMOV.32.CLEAR B10, B9 ;                        /* 0x00000000090a7f55 */
                                                                                  /* 0x000fe20000100000 */
        /*01e0*/                   BMOV.32.CLEAR B11, B10 ;                       /* 0x000000000a0b7f55 */
                                                                                  /* 0x000fe20000100000 */
        /*01f0*/                   BMOV.32.CLEAR B12, B11 ;                       /* 0x000000000b0c7f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0200*/                   BMOV.32.CLEAR B13, B12 ;                       /* 0x000000000c0d7f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0210*/                   BMOV.32.CLEAR B14, B13 ;                       /* 0x000000000d0e7f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0220*/                   BMOV.32.CLEAR B15, B14 ;                       /* 0x000000000e0f7f55 */
                                                                                  /* 0x000fe20000100000 */
        /*0230*/                   BMOV.32 B15, 0x0 ;                             /* 0x000000000f007956 */
                                                                                  /* 0x000fe20000000000 */
        /*0240*/                   UMOV UR2, 0x10 ;                               /* 0x0000001000027882 */
                                                                                  /* 0x000fe20000000000 */
        /*0250*/                   STL.128 [0xffffb0], R8 ;                       /* 0xffffb008ff007387 */
                                                                                  /* 0x0103e20000100c00 */
        /*0260*/                   BMOV.32 R0, TRAP_RETURN_PC.LO ;                /* 0x0000000015007355 */
                                                                                  /* 0x004f220000000000 */
        /*0270*/                   BMOV.32 R1, TRAP_RETURN_PC.HI ;                /* 0x0000000016017355 */
                                                                                  /* 0x004e240000000000 */
        /*0280*/                   STL.64 [0xffffa8], R0 ;                        /* 0xffffa800ff007387 */
                                                                                  /* 0x0113e40000100a00 */
        /*0290*/                   MOV R1, 0xffffa0 ;                             /* 0x00ffffa000017802 */
                                                                                  /* 0x002fe20000000f00 */
        /*02a0*/                   S2R R2, SR_GLOBALERRORSTATUS ;                 /* 0x0000000000027919 */
                                                                                  /* 0x004e240000004000 */
        /*02b0*/                   LOP3.LUT R2, R2, 0x4, RZ, 0xc0, !PT ;          /* 0x0000000402027812 */
                                                                                  /* 0x001fec00078ec0ff */
        /*02c0*/                   ISETP.EQ.AND P1, PT, R2, 0x4, PT ;             /* 0x000000040200780c */
                                                                                  /* 0x000fda0003f22270 */
        /*02d0*/              @!P1 BRA.U 0x3d0 ;                                  /* 0x000000f100009947 */
                                                                                  /* 0x000fea0003800000 */
        /*02e0*/                   MOV R2, 0x1e000000 ;                           /* 0x1e00000000027802 */
                                                                                  /* 0x000fe40000000f00 */
        /*02f0*/                   MOV R3, 0x7fac ;                               /* 0x00007fac00037802 */
                                                                                  /* 0x004fe20000000f00 */
        /*0300*/                   S2R R4, SR_VIRTUALSMID ;                       /* 0x0000000000047919 */
                                                                                  /* 0x010e220000004300 */
        /*0310*/                   MOV R5, 0x74a20 ;                              /* 0x00074a2000057802 */
                                                                                  /* 0x010fec0000000f00 */
        /*0320*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                 /* 0x0000000504027225 */
                                                                                  /* 0x001fea00078e0002 */
        /*0330*/                   IADD3 R2, P3, R2, 0x69020, RZ ;                /* 0x0006902002027810 */
                                                                                  /* 0x000fea0007f7e0ff */
        /*0340*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;              /* 0x000000ff03037210 */
                                                                                  /* 0x000fe20001ffe4ff */
        /*0350*/                   S2R R4, SR_VIRTID ;                            /* 0x0000000000047919 */
                                                                                  /* 0x000e240000000300 */
        /*0360*/                   SHF.R.U32.HI R4, RZ, 0x8, R4 ;                 /* 0x00000008ff047819 */
                                                                                  /* 0x001fec0000011604 */
        /*0370*/                   SGXT.U32 R4, R4, 0x7 ;                         /* 0x000000070404781a */
                                                                                  /* 0x000fe40000000000 */
        /*0380*/                   MOV R5, 0x2c0 ;                                /* 0x000002c000057802 */
                                                                                  /* 0x000fec0000000f00 */
        /*0390*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                 /* 0x0000000504027225 */
                                                                                  /* 0x000fea00078e0002 */
        /*03a0*/                   LD.E.64 R4, [R2+0x288] ;                       /* 0x0000028802047980 */
                                                                                  /* 0x0005220000100b00 */
        /*03b0*/                   CCTLL.IVALL ;                                  /* 0x00000000ff007990 */
                                                                                  /* 0x000fe20002000000 */
        /*03c0*/                   SETLMEMBASE R4 ;                               /* 0x00000000040073c1 */
                                                                                  /* 0x0109c00000000000 */
        /*03d0*/                   R2P PR, RZ ;                                   /* 0x000000ffff007804 */
                                                                                  /* 0x000fe40000000000 */
// trap_entry_bootstrap end


        /*03e0*/                   IADD3 R1, R1, -0x10, RZ ;                      /* 0xfffffff001017810 */
                                                                                  /* 0x002fe40007ffe0ff */
        /*03f0*/                   MOV R2, RZ ;                                   /* 0x000000ff00027202 */
                                                                                  /* 0x004fe20000000f00 */

// TRAP_ABI_CALL(do_read_trap_reason) (0x7fac32f6ac00)
        /*0400*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*0410*/                   CALL.ABS 0x7fac32f6ac00;                       /* 0x32f6ac0000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*0420*/                   MOV R3, RZ ;                                   /* 0x000000ff00037202 */
                                                                                  /* 0x005fea0000000f00 */
        /*0430*/                   STL.64 [R1], R2 ;                              /* 0x0000000201007387 */
                                                                                  /* 0x0085e20000100a00 */

// check_preempt_restore:
        /*0440*/                   LOP3.LUT P0, RZ, R2, 0x1, RZ, 0xc0, !PT ;      /* 0x0000000102ff7812 */
                                                                                  /* 0x000fda000780c0ff */
        /*0450*/              @!P0 BRA.U 0x490 ;                                  /* 0x0000003100008947 */
                                                                                  /* 0x000fea0003800000 */
// TRAP_ABI_CALL(do_preempt_restore) (0x7fac32f6be00)
        /*0460*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*0470*/                   CALL.ABS 0x7fac32f6be00;                       /* 0x32f6be0000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*0480*/                   LDL R2, [R1] ;                                 /* 0x0000000001027983 */
                                                                                  /* 0x0044e40000100800 */


// check_detect_continuation:
        /*0490*/                   LOP3.LUT P0, RZ, R2, 0x2, RZ, 0xc0, !PT ;      /* 0x0000000202ff7812 */
                                                                                  /* 0x008fda000780c0ff */
        /*04a0*/              @!P0 BRA.U 0x500 ;                                  /* 0x0000005100008947 */
                                                                                  /* 0x000fea0003800000 */
// will not be executed here, branch out to 0x500
// TRAP_ABI_CALL(do_detect_continuation) (0x7fac32f6c900)
        /*04b0*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*04c0*/                   CALL.ABS 0x7fac32f6c900;                       /* 0x32f6c90000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*04d0*/                   MOV R3, R2 ;                                   /* 0x0000000200037202 */
                                                                                  /* 0x005fea0000000f00 */
        /*04e0*/                   STL [R1+0x4], R3 ;                             /* 0x0000040301007387 */
                                                                                  /* 0x0005e40000100800 */
        /*04f0*/                   LDL R2, [R1] ;                                 /* 0x0000000001027983 */
                                                                                  /* 0x0044e40000100800 */


// check_set_defer_continuation:
        /*0500*/                   LOP3.LUT P0, RZ, R2, 0x4, RZ, 0xc0, !PT ;      /* 0x0000000402ff7812 */
                                                                                  /* 0x008fda000780c0ff */
        /*0510*/              @!P0 BRA.U 0x550 ;                                  /* 0x0000003100008947 */
                                                                                  /* 0x000fea0003800000 */
// will not be executed here, branch out to 0x550
// TRAP_ABI_CALL(do_set_defer_continuations) (0x7fac32f6d300)
        /*0520*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*0530*/                   CALL.ABS 0x7fac32f6d300;                       /* 0x32f6d30000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*0540*/                   LDL R2, [R1] ;                                 /* 0x0000000001027983 */
                                                                                  /* 0x0044e40000100800 */
                                                                

// check_pause:
        /*0550*/                   LOP3.LUT P0, RZ, R2, 0x8, RZ, 0xc0, !PT ;      /* 0x0000000802ff7812 */
                                                                                  /* 0x008fda000780c0ff */
        /*0560*/              @!P0 BRA.U 0x5b0 ;                                  /* 0x0000004100008947 */
                                                                                  /* 0x000fea0003800000 */
// TRAP_ABI_CALL(do_pause) (0x7fac32f6de00)
        /*0570*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*0580*/                   CALL.ABS 0x7fac32f6de00;                       /* 0x32f6de0000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*0590*/                   MOV R2, RZ ;                                   /* 0x000000ff00027202 */
                                                                                  /* 0x004fe20000000f00 */
        /*05a0*/                   BRA.U 0x400 ;                                  /* 0xfffffe5100007947 */
                                                                                  /* 0x000fea000383ffff */


// check_preempt_save:
        /*05b0*/                   LOP3.LUT P0, RZ, R2, 0x10, RZ, 0xc0, !PT ;     /* 0x0000001002ff7812 */
                                                                                  /* 0x008fda000780c0ff */
        /*05c0*/              @!P0 BRA.U 0x610 ;                                  /* 0x0000004100008947 */
                                                                                  /* 0x000fea0003800000 */
// TRAP_ABI_CALL(do_preempt_save) (0x7fac32f6f000)
        /*05d0*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*05e0*/                   CALL.ABS 0x7fac32f6f000;                       /* 0x32f6f00000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*05f0*/                   LDL R2, [R1] ;                                 /* 0x0000000001027983 */
                                                                                  /* 0x004fe20000100800 */
        /*0600*/                   BRA.U 0x840 ;                                  /* 0x0000023100007947 */
                                                                                  /* 0x000fea0003800000 */


// check_warp_is_sw_paused:
        /*0610*/                   LOP3.LUT P0, RZ, R2, 0x20, RZ, 0xc0, !PT ;     /* 0x0000002002ff7812 */
                                                                                  /* 0x008fda000780c0ff */
        /*0620*/              @!P0 BRA.U 0x670 ;                                  /* 0x0000004100008947 */
                                                                                  /* 0x000fea0003800000 */
// will not be executed here, branch out to 0x670
// TRAP_ABI_CALL(do_pause_quiet) (0x7fac32f6fb00)
        /*0630*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*0640*/                   CALL.ABS 0x7fac32f6fb00;                       /* 0x32f6fb0000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*0650*/                   MOV R2, RZ ;                                   /* 0x000000ff00027202 */
                                                                                  /* 0x004fe20000000f00 */
        /*0660*/                   BRA.U 0x400 ;                                  /* 0xfffffd9100007947 */
                                                                                  /* 0x000fea000383ffff */


// check_warp_has_defer_continuations:
        /*0670*/                   LOP3.LUT P0, RZ, R2, 0x40, RZ, 0xc0, !PT ;     /* 0x0000004002ff7812 */
                                                                                  /* 0x008fda000780c0ff */
        /*0680*/              @!P0 BRA.U 0x6e0 ;                                  /* 0x0000005100008947 */
                                                                                  /* 0x000fea0003800000 */
// will not be executed here, branch out to 0x6e0
// TRAP_ABI_CALL(get_defer_continuation_state_and_clear) (0x7fac32f70500)
        /*0690*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*06a0*/                   CALL.ABS 0x7fac32f70500;                       /* 0x32f7050000007943 */
                                                                                  /* 0x000fc00003807fac */
        /*06b0*/                   MOV R3, R2 ;                                   /* 0x0000000200037202 */
                                                                                  /* 0x005fea0000000f00 */
        /*06c0*/                   STL [R1+0x4], R3 ;                             /* 0x0000040301007387 */
                                                                                  /* 0x0005e40000100800 */
        /*06d0*/                   LDL R2, [R1] ;                                 /* 0x0000000001027983 */
                                                                                  /* 0x0044e40000100800 */


// check_handle_continuations:
        /*06e0*/                   LOP3.LUT P0, RZ, R2, 0x80, RZ, 0xc0, !PT ;     /* 0x0000008002ff7812 */
                                                                                  /* 0x008fda000780c0ff */
        /*06f0*/              @!P0 BRA.U 0x730 ;                                  /* 0x0000003100008947 */
                                                                                  /* 0x000fea0003800000 */
// will not be executed here, branch out to 0x730
// TRAP_ABI_CALL(do_handle_continuations) (0x7fac32f71000)
        /*0700*/                   LDL R3, [R1+0x4] ;                             /* 0x0000040001037983 */
                                                                                  /* 0x0054220000100800 */
        /*0710*/                   LEPC R12 ;                                     /* 0x00000000000c734e */
                                                                                  /* 0x000fe20000000000 */
        /*0720*/                   CALL.ABS 0x7fac32f71000;                       /* 0x32f7100000007943 */
                                                                                  /* 0x000fc00003807fac */

// trap_exit_bootstrap begin
        /*0730*/                   LDL.64 R4, [0xffffa8] ;                        /* 0xffffa800ff047983 */
                                                                                  /* 0x010e640000100a00 */
        /*0740*/                   BMOV.32 TRAP_RETURN_PC.LO, R4 ;                /* 0x0000000415007356 */
                                                                                  /* 0x0023e20000000000 */
        /*0750*/                   BMOV.32 TRAP_RETURN_PC.HI, R5 ;                /* 0x0000000516007356 */
                                                                                  /* 0x000be20000000000 */
        /*0760*/                   LDL.128 R8, [0xffffb0] ;                       /* 0xffffb000ff087983 */
                                                                                  /* 0x012e640000100c00 */
        /*0770*/                   RPCMOV.32 Rpc.LO, R8 ;                         /* 0x0000000800007352 */
                                                                                  /* 0x002fec0000000000 */
        /*0780*/                   RPCMOV.32 Rpc.HI, R9 ;                         /* 0x0000000980007352 */
                                                                                  /* 0x000fec0000000000 */
        /*0790*/                   BMOV.32 OPT_STACK, R10 ;                       /* 0x0000000a1c007356 */
                                                                                  /* 0x0009e20000000000 */
        /*07a0*/                   BMOV.32 B0, R11 ;                              /* 0x0000000b00007356 */
                                                                                  /* 0x0003e40000000000 */
        /*07b0*/                   LDL.128 R0, [0xfffff0] ;                       /* 0xfffff000ff007983 */
                                                                                  /* 0x01fe240000100c00 */
        /*07c0*/                   MOV R12, R0 ;                                  /* 0x00000000000c7202 */
                                                                                  /* 0x001fe40000000f00 */
        /*07d0*/                   MOV R13, R1 ;                                  /* 0x00000001000d7202 */
                                                                                  /* 0x000fe40000000f00 */
        /*07e0*/                   R2P PR, R2 ;                                   /* 0x000000ff02007804 */
                                                                                  /* 0x000fe20000000000 */
        /*07f0*/                   BMOV.32 TRAP_RETURN_MASK, R3 ;                 /* 0x0000000317007356 */
                                                                                  /* 0x0001e20000000000 */
        /*0800*/                   LDL.128 R8, [0xffffe0] ;                       /* 0xffffe000ff087983 */
                                                                                  /* 0x012fe20000100c00 */
        /*0810*/                   LDL.128 R4, [0xffffd0] ;                       /* 0xffffd000ff047983 */
                                                                                  /* 0x032fe20000100c00 */
        /*0820*/                   LDL.128 R0, [0xffffc0] ;                       /* 0xffffc000ff007983 */
                                                                                  /* 0x001fe20000100c00 */
// trap_exit_bootstrap end
        /*0830*/                   RTT;                                           /* 0x000000000000794f */
                                                                                  /* 0x000fc00000000000 */
        /*0840*/                   EXIT.PREEMPTED.NO_ATEXIT;                      /* 0x000000000000794d */
                                                                                  /* 0x000fc00003e00000 */
        /*0850*/                   BRA 0x850;                                     /* 0xfffffff000007947 */
                                                                                  /* 0x000fc0000383ffff */
        /*0860*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*0870*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*0880*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*0890*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*08a0*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*08b0*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*08c0*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*08d0*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*08e0*/                   NOP;                                           /* 0x0000000000007918 */
                                                                                  /* 0x000fc00000000000 */
        /*08f0*/                   NOP;                                           /* 0x0000000000007918 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f6ac00
// function name: do_read_trap_reason

	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*0010*/                   MOV R2, RZ ;                                 /* 0x000000ff00027202 */
                                                                                /* 0x000fe40000000f00 */
        /*0020*/                   ISETP.NE.AND P6, PT, RZ, RZ, PT ;            /* 0x000000ffff00720c */
                                                                                /* 0x000fe20003fc5270 */
// GET_WARP_SCRATCHPAD_FIELD_8(warpFlagsPauseServiced, R4, R5, R0, P0) begin
// GET_WARP_SCRATCHPAD_ADDR(R4, R5, R0, P0) begin
        /*0030*/                   S2R R4, SR_VIRTUALSMID ;                     /* 0x0000000000047919 */
                                                                                /* 0x000e220000004300 */
        /*0040*/                   S2R R5, SR_VIRTID ;                          /* 0x0000000000057919 */
                                                                                /* 0x000e640000000300 */
        /*0050*/                   SHF.R.U32.HI R5, RZ, 0x8, R5 ;               /* 0x00000008ff057819 */
                                                                                /* 0x002fec0000011605 */
        /*0060*/                   SGXT.U32 R5, R5, 0x7 ;                       /* 0x000000070505781a */
                                                                                /* 0x000fec0000000000 */
        /*0070*/                   IMAD.U32 R0, R4, 0x40, R5 ;                  /* 0x0000004004007824 */
                                                                                /* 0x001fec00078e0005 */
        /*0080*/                   IMAD R0, R0, 0x198, RZ ;                     /* 0x0000019800007824 */
                                                                                /* 0x000fe200078e02ff */
        /*0090*/                   MOV R4, 0x33400000 ;                         /* 0x3340000000047802 */
                                                                                /* 0x000fe80000000f00 */
        /*00a0*/                   MOV R5, 0x7fac ;                             /* 0x00007fac00057802 */
                                                                                /* 0x000fe40000000f00 */
        /*00b0*/                   IADD3 R4, P0, R4, 0x10, RZ ;                 /* 0x0000001004047810 */
                                                                                /* 0x000fea0007f1e0ff */
        /*00c0*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x000fe400007fe4ff */
        /*00d0*/                   IADD3 R4, P0, R4, R0, RZ ;                   /* 0x0000000004047210 */
                                                                                /* 0x000fea0007f1e0ff */
        /*00e0*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x000fea00007fe4ff */
// GET_WARP_SCRATCHPAD_ADDR(R4, R5, R0, P0) end

        /*00f0*/                   LD.E.U8.CONSTANT R0, [R4+0x195] ;            /* 0x0000019504007980 */
                                                                                /* 0x0000640000108100 */
// GET_WARP_SCRATCHPAD_FIELD_8(warpFlagsPauseServiced, R4, R5, R0, P0) end


        /*0100*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;            /* 0x000000ff0000720c */
                                                                                /* 0x002fda0003f05270 */

// SET_THSTATE_BIT_ON(P0, R2, WARP_IS_SW_PAUSED) begin
        /*0110*/               @P0 LOP3.LUT R2, R2, 0x20, RZ, 0xfc, !PT ;       /* 0x0000002002020812 */
                                                                                /* 0x000fe400078efcff */
// SET_THSTATE_BIT_ON(P0, R2, WARP_IS_SW_PAUSED) end


// GET_CILP_BUFFER_BASE_SM begin
// GET_CILP_BUFFER_BASE begin
        /*0120*/                   MOV R4, 0x1e000000 ;                         /* 0x1e00000000047802 */
                                                                                /* 0x001fe40000000f00 */
        /*0130*/                   MOV R5, 0x7fac ;                             /* 0x00007fac00057802 */
                                                                                /* 0x001fe20000000f00 */
// GET_CILP_BUFFER_BASE end
// GET_SM_ID begin
        /*0140*/                   S2R R0, SR_VIRTUALSMID ;                     /* 0x0000000000007919 */
                                                                                /* 0x000e620000004300 */
// GET_SM_ID end

        /*0150*/                   MOV R3, 0x74a20 ;                            /* 0x00074a2000037802 */
                                                                                /* 0x000fec0000000f00 */
// SCALED_OFFSET_FROM_BASE begin
        /*0160*/                   IMAD.WIDE.U32 R4, R0, R3, R4 ;               /* 0x0000000300047225 */
                                                                                /* 0x003fea00078e0004 */
// SCALED_OFFSET_FROM_BASE end

        /*0170*/                   IADD3 R4, P0, R4, 0x8, RZ ;                  /* 0x0000000804047810 */
                                                                                /* 0x001fea0007f1e0ff */
        /*0180*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x001fe200007fe4ff */
        /*0190*/                   S2R R0, SR_VIRTID ;                          /* 0x0000000000007919 */
                                                                                /* 0x000e640000000300 */
        /*01a0*/                   SHF.R.U32.HI R0, RZ, 0x8, R0 ;               /* 0x00000008ff007819 */
                                                                                /* 0x002fec0000011600 */
        /*01b0*/                   SGXT.U32 R0, R0, 0x7 ;                       /* 0x000000070000781a */
                                                                                /* 0x000fec0000000000 */
        /*01c0*/                   ISETP.GE.U32.AND P0, PT, R0, 0x20, PT ;      /* 0x000000200000780c */
                                                                                /* 0x000fda0003f06070 */
        /*01d0*/               @P0 IADD3 R4, P1, R4, 0x4, RZ ;                  /* 0x0000000404040810 */
                                                                                /* 0x001fea0007f3e0ff */
        /*01e0*/               @P0 IADD3.X R5, R5, RZ, RZ, P1, !PT ;            /* 0x000000ff05050210 */
                                                                                /* 0x001fe40000ffe4ff */
        /*01f0*/               @P0 IADD3 R0, R0, -0x20, RZ ;                    /* 0xffffffe000000810 */
                                                                                /* 0x000fec0007ffe0ff */
        /*0200*/                   SHF.L.U32.HI R0, RZ, R0, 0x1 ;               /* 0x00000001ff007419 */
                                                                                /* 0x000fe80000010600 */
        /*0210*/                   LD.E.CONSTANT R3, [R4] ;                     /* 0x0000000004037980 */
                                                                                /* 0x0000640000108900 */
        /*0220*/                   LOP3.LUT P0, R3, R3, R0, RZ, 0xc0, !PT ;     /* 0x0000000003037212 */
                                                                                /* 0x002fda000780c0ff */
        /*0230*/               @P0 LOP3.LUT R2, R2, 0x40, RZ, 0xfc, !PT ;       /* 0x0000004002020812 */
                                                                                /* 0x000fe200078efcff */
        /*0240*/                   S2R R0, SR_GLOBALERRORSTATUS ;               /* 0x0000000000007919 */
                                                                                /* 0x000e640000004000 */
        /*0250*/                   R2P PR, R0 ;                                 /* 0x000000ff00007804 */
                                                                                /* 0x002fda0000000000 */
        /*0260*/               @P1 LOP3.LUT R2, R2, 0x10, RZ, 0xfc, !PT ;       /* 0x0000001002021812 */
                                                                                /* 0x000fec00078efcff */
        /*0270*/               @P2 LOP3.LUT R2, R2, 0x1, RZ, 0xfc, !PT ;        /* 0x0000000102022812 */
                                                                                /* 0x000fe800078efcff */
        /*0280*/                   R2P PR, RZ ;                                 /* 0x000000ffff007804 */
                                                                                /* 0x000fe40000000000 */
        /*0290*/                   LOP3.LUT P0, RZ, R2, 0x20, RZ, 0xc0, !PT ;   /* 0x0000002002ff7812 */
                                                                                /* 0x000fe4000780c0ff */
        /*02a0*/                   LOP3.LUT P5, RZ, R2, 0x1, RZ, 0xc0, !PT ;    /* 0x0000000102ff7812 */
                                                                                /* 0x000fea00078ac0ff */
        /*02b0*/                   PLOP3.LUT P0, PT, P0, P5, PT, 0xf8, 0x8f ;   /* 0x00000000008f781c */
                                                                                /* 0x000ff2000070bf70 */
        /*02c0*/                   MOV R0, RZ ;                                 /* 0x000000ff00007202 */
                                                                                /* 0x000fe80000000f00 */
        /*02d0*/              @!P0 S2R R0, SR_WARPERRORSTATUS ;                 /* 0x0000000000008919 */
                                                                                /* 0x000e640000004200 */
        /*02e0*/                   LOP3.LUT P1, RZ, R0, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff00ff7812 */
                                                                                /* 0x002fe4000782c0ff */
        /*02f0*/                   SHF.R.U32.HI R3, RZ, 0x18, R0 ;              /* 0x00000018ff037819 */
                                                                                /* 0x000fec0000011600 */
        /*0300*/                   SGXT.U32 R3, R3, 0x3 ;                       /* 0x000000030303781a */
                                                                                /* 0x000fec0000000000 */
        /*0310*/                   ISETP.EQ.U32.AND P2, PT, R3, 0x7, PT ;       /* 0x000000070300780c */
                                                                                /* 0x000fe40003f42070 */
        /*0320*/                   ISETP.EQ.U32.AND P3, PT, R3, 0x3, PT ;       /* 0x000000030300780c */
                                                                                /* 0x000fe40003f62070 */
        /*0330*/                   ISETP.EQ.U32.AND P4, PT, R3, 0x1, PT ;       /* 0x000000010300780c */
                                                                                /* 0x000fe40003f82070 */
        /*0340*/                   MOV R4, 0x33400000 ;                         /* 0x3340000000047802 */
                                                                                /* 0x001fe40000000f00 */
        /*0350*/                   MOV R5, 0x7fac ;                             /* 0x00007fac00057802 */
                                                                                /* 0x001fea0000000f00 */
        /*0360*/                   LD.E.CONSTANT R0, [R4+0x4] ;                 /* 0x0000000404007980 */
                                                                                /* 0x0000a40000108900 */
        /*0370*/                   ISETP.EQ.AND P5, PT, R0, 0x1, P5 ;           /* 0x000000010000780c */
                                                                                /* 0x004fe40002fa2270 */
        /*0380*/                   MOV R7, 0x290 ;                              /* 0x0000029000077802 */
                                                                                /* 0x000fe20000000f00 */
        /*0390*/                   S2R R0, SR_GLOBALERRORSTATUS ;               /* 0x0000000000007919 */
                                                                                /* 0x000ea40000004000 */
        /*03a0*/                   LOP3.LUT P0, RZ, R0, 0x1, RZ, 0xc0, !PT ;    /* 0x0000000100ff7812 */
                                                                                /* 0x004fda000780c0ff */
        /*03b0*/              @!P0 BRA.U 0x520 ;                                /* 0x0000016100008947 */
                                                                                /* 0x000fea0003800000 */
        /*03c0*/                   LOP3.LUT P0, RZ, R2, 0x1, RZ, 0xc0, !PT ;    /* 0x0000000102ff7812 */
                                                                                /* 0x000fda000780c0ff */
        /*03d0*/               @P0 BRA.U 0x520 ;                                /* 0x0000014100000947 */
                                                                                /* 0x000fea0003800000 */

// GET_WARP_SCRATCHPAD_FIELD_32()
// GET_WARP_SCRATCHPAD_ADDR(R4, R5, R0, P0) begin
        /*03e0*/                   S2R R4, SR_VIRTUALSMID ;                     /* 0x0000000000047919 */
                                                                                /* 0x001ea20000004300 */
        /*03f0*/                   S2R R5, SR_VIRTID ;                          /* 0x0000000000057919 */
                                                                                /* 0x001ee40000000300 */
        /*0400*/                   SHF.R.U32.HI R5, RZ, 0x8, R5 ;               /* 0x00000008ff057819 */
                                                                                /* 0x009fec0000011605 */
        /*0410*/                   SGXT.U32 R5, R5, 0x7 ;                       /* 0x000000070505781a */
                                                                                /* 0x001fec0000000000 */
        /*0420*/                   IMAD.U32 R0, R4, 0x40, R5 ;                  /* 0x0000004004007824 */
                                                                                /* 0x004fec00078e0005 */
        /*0430*/                   IMAD R0, R0, 0x198, RZ ;                     /* 0x0000019800007824 */
                                                                                /* 0x000fe200078e02ff */
        /*0440*/                   MOV R4, 0x33400000 ;                         /* 0x3340000000047802 */
                                                                                /* 0x001fe80000000f00 */
        /*0450*/                   MOV R5, 0x7fac ;                             /* 0x00007fac00057802 */
                                                                                /* 0x001fe40000000f00 */
        /*0460*/                   IADD3 R4, P0, R4, 0x10, RZ ;                 /* 0x0000001004047810 */
                                                                                /* 0x001fea0007f1e0ff */
        /*0470*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x001fe400007fe4ff */
        /*0480*/                   IADD3 R4, P0, R4, R0, RZ ;                   /* 0x0000000004047210 */
                                                                                /* 0x001fea0007f1e0ff */
        /*0490*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x001fea00007fe4ff */
// GET_WARP_SCRATCHPAD_ADDR(R4, R5, R0, P0) end

        /*04a0*/                   LD.E.CONSTANT R0, [R4+0x17c] ;               /* 0x0000017c04007980 */
                                                                                /* 0x0000a40000108900 */
// GET_WARP_SCRATCHPAD_FIELD_32() end


        /*04b0*/                   ISETP.EQ.AND P0, PT, R0, RZ, PT ;            /* 0x000000ff0000720c */
                                                                                /* 0x004fda0003f02270 */
        /*04c0*/              @!P0 IADD3 R0, R0, -0x1, RZ ;                     /* 0xffffffff00008810 */
                                                                                /* 0x000fe20007ffe0ff */
        /*04d0*/                   S2R R3, SR_LANEID ;                          /* 0x0000000000037919 */
                                                                                /* 0x000e640000000000 */
        /*04e0*/                   ISETP.EQ.AND P0, PT, R3, RZ, PT ;            /* 0x000000ff0300720c */
                                                                                /* 0x002fda0003f02270 */
        /*04f0*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x17c], R0 ;      /* 0x0000017c04000385 */
                                                                                /* 0x0001e20000108900 */
        /*0500*/                   ISETP.EQ.AND P0, PT, R0, RZ, PT ;            /* 0x000000ff0000720c */
                                                                                /* 0x000fda0003f02270 */
        /*0510*/               @P0 LOP3.LUT R7, R7, 0x1, RZ, 0xfc, !PT ;        /* 0x0000000107070812 */
                                                                                /* 0x000fe800078efcff */
        /*0520*/                   S2R R0, SR_GLOBALERRORSTATUS ;               /* 0x0000000000007919 */
                                                                                /* 0x005ea40000004000 */
        /*0530*/                   LOP3.LUT P0, RZ, R0, R7, RZ, 0xc0, !PT ;     /* 0x0000000700ff7212 */
                                                                                /* 0x004fda000780c0ff */
        /*0540*/                   P2R R0, PR, RZ, 0x3f ;                       /* 0x0000003fff007803 */
                                                                                /* 0x000fec0000000000 */
        /*0550*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;            /* 0x000000ff0000720c */
                                                                                /* 0x000ff00003f05270 */
        /*0560*/                   ISETP.NE.AND P6, PT, RZ, RZ, PT ;            /* 0x000000ffff00720c */
                                                                                /* 0x000fe20003fc5270 */
        /*0570*/                   B2R R6, 0x0 ;                                /* 0x000000000006731c */
                                                                                /* 0x000ea20000000000 */
        /*0580*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*0590*/                   R2B 0x0, RZ ;                                /* 0x000000ff0000731e */
                                                                                /* 0x000fe20000000000 */
        /*05a0*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*05b0*/               @P0 MOV R7, 0x80000000 ;                         /* 0x8000000000070802 */
                                                                                /* 0x000fea0000000f00 */
        /*05c0*/               @P0 R2B 0x0, R7 ;                                /* 0x000000070000031e */
                                                                                /* 0x0009e20000000000 */
        /*05d0*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*05e0*/                   B2R R7, 0x0 ;                                /* 0x000000000007731c */
                                                                                /* 0x010ee40000000000 */
        /*05f0*/                   ISETP.NE.AND P0, PT, R7, RZ, PT ;            /* 0x000000ff0700720c */
                                                                                /* 0x008ff40003f05270 */
        /*0600*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*0610*/                   R2B 0x0, R6 ;                                /* 0x000000060000731e */
                                                                                /* 0x0045e20000000000 */
        /*0620*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*0630*/               @P0 LOP3.LUT R2, R2, 0x8, RZ, 0xfc, !PT ;        /* 0x0000000802020812 */
                                                                                /* 0x000fec00078efcff */
        /*0640*/                   LOP3.LUT P0, RZ, R2, 0x40, RZ, 0xc0, !PT ;   /* 0x0000004002ff7812 */
                                                                                /* 0x000fea000780c0ff */
        /*0650*/                   PLOP3.LUT P0, PT, P0, P6, PT, 0xf8, 0x8f ;   /* 0x00000000008f781c */
                                                                                /* 0x000ff2000070df70 */
        /*0660*/                   MOV R0, RZ ;                                 /* 0x000000ff00007202 */
                                                                                /* 0x000fe80000000f00 */
        /*0670*/              @!P0 S2R R0, SR_WARPERRORSTATUS ;                 /* 0x0000000000008919 */
                                                                                /* 0x000ee40000004200 */
        /*0680*/                   SHF.R.U32.HI R3, RZ, 0x18, R0 ;              /* 0x00000018ff037819 */
                                                                                /* 0x00afec0000011600 */
        /*0690*/                   SGXT.U32 R3, R3, 0x3 ;                       /* 0x000000030303781a */
                                                                                /* 0x000fec0000000000 */
        /*06a0*/                   ISETP.EQ.U32.AND P0, PT, R3, 0x2, PT ;       /* 0x000000020300780c */
                                                                                /* 0x000ff20003f02070 */
        /*06b0*/                   B2R R6, 0x0 ;                                /* 0x000000000006731c */
                                                                                /* 0x004e620000000000 */
        /*06c0*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*06d0*/                   R2B 0x0, RZ ;                                /* 0x000000ff0000731e */
                                                                                /* 0x000fe20000000000 */
        /*06e0*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*06f0*/               @P0 MOV R7, 0x80000000 ;                         /* 0x8000000000070802 */
                                                                                /* 0x010fea0000000f00 */
        /*0700*/               @P0 R2B 0x0, R7 ;                                /* 0x000000070000031e */
                                                                                /* 0x0009e20000000000 */
        /*0710*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*0720*/                   B2R R7, 0x0 ;                                /* 0x000000000007731c */
                                                                                /* 0x010ea40000000000 */
        /*0730*/                   ISETP.NE.AND P0, PT, R7, RZ, PT ;            /* 0x000000ff0700720c */
                                                                                /* 0x004ff40003f05270 */
        /*0740*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*0750*/                   R2B 0x0, R6 ;                                /* 0x000000060000731e */
                                                                                /* 0x002fe20000000000 */
        /*0760*/                   BAR.SYNCALL ;                                /* 0x0000000000007b1d */
                                                                                /* 0x000fec0000008000 */
        /*0770*/               @P0 LOP3.LUT R2, R2, 0x2, RZ, 0xfc, !PT ;        /* 0x0000000202020812 */
                                                                                /* 0x000fe400078efcff */
        /*0780*/                   MOV R3, 0x41 ;                               /* 0x0000004100037802 */
                                                                                /* 0x000fec0000000f00 */
        /*0790*/                   LOP3.LUT P0, R0, R2, R3, RZ, 0xc0, !PT ;     /* 0x0000000302007212 */
                                                                                /* 0x000fec000780c0ff */
        /*07a0*/                   ISETP.EQ.AND P0, PT, R0, R3, PT ;            /* 0x000000030000720c */
                                                                                /* 0x000fda0003f02270 */
        /*07b0*/               @P0 LOP3.LUT R2, R2, 0x80, RZ, 0xfc, !PT ;       /* 0x0000008002020812 */
                                                                                /* 0x000fec00078efcff */
        /*07c0*/                   LOP3.LUT P0, RZ, R2, 0x2, RZ, 0xc0, !PT ;    /* 0x0000000202ff7812 */
                                                                                /* 0x000fe4000780c0ff */
        /*07d0*/                   MOV R3, 0x18 ;                               /* 0x0000001800037802 */
                                                                                /* 0x000fec0000000f00 */
        /*07e0*/                   LOP3.LUT P1, R0, R2, R3, RZ, 0xc0, !PT ;     /* 0x0000000302007212 */
                                                                                /* 0x000fec000782c0ff */
        /*07f0*/                   ISETP.NE.AND P1, PT, R0, RZ, PT ;            /* 0x000000ff0000720c */
                                                                                /* 0x000fea0003f25270 */
        /*0800*/                   PLOP3.LUT P1, PT, P0, P1, PT, 0x80, 0x8 ;    /* 0x000000000008781c */
                                                                                /* 0x000fe40000723070 */
        /*0810*/                   MOV R3, 0x80 ;                               /* 0x0000008000037802 */
                                                                                /* 0x000fe40000000f00 */
        /*0820*/                   MOV R4, 0x4 ;                                /* 0x0000000400047802 */
                                                                                /* 0x001fec0000000f00 */
        /*0830*/               @P0 SEL R3, R4, R3, P1 ;                         /* 0x0000000304030207 */
                                                                                /* 0x000fec0000800000 */
        /*0840*/               @P0 LOP3.LUT R2, R2, R3, RZ, 0xfc, !PT ;         /* 0x0000000302020212 */
                                                                                /* 0x000fe200078efcff */
        /*0850*/                   RET.ABS R12 0x20;                            /* 0x000000200c007950 */
                                                                                /* 0x000fc00003a00000 */
        /*0860*/                   BRA 0x860;                                   /* 0xfffffff000007947 */
                                                                                /* 0x000fc0000383ffff */
        /*0870*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0880*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0890*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*08a0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*08b0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*08c0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*08d0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*08e0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*08f0*/                   NOP;                                         /* 0x0000000000007918 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f6be00
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   S2R R4, SR_VIRTUALSMID ;                     /* 0x0000000000047919 */
                                                                                /* 0x000e220000004300 */
        /*0010*/                   S2R R5, SR_VIRTID ;                          /* 0x0000000000057919 */
                                                                                /* 0x000e640000000300 */
        /*0020*/                   SHF.R.U32.HI R5, RZ, 0x8, R5 ;               /* 0x00000008ff057819 */
                                                                                /* 0x002fec0000011605 */
        /*0030*/                   SGXT.U32 R5, R5, 0x7 ;                       /* 0x000000070505781a */
                                                                                /* 0x000fec0000000000 */
        /*0040*/                   IMAD.U32 R0, R4, 0x40, R5 ;                  /* 0x0000004004007824 */
                                                                                /* 0x001fec00078e0005 */
        /*0050*/                   IMAD R0, R0, 0x198, RZ ;                     /* 0x0000019800007824 */
                                                                                /* 0x000fe200078e02ff */
        /*0060*/                   MOV R4, 0x33400000 ;                         /* 0x3340000000047802 */
                                                                                /* 0x000fe80000000f00 */
        /*0070*/                   MOV R5, 0x7fac ;                             /* 0x00007fac00057802 */
                                                                                /* 0x000fe40000000f00 */
        /*0080*/                   IADD3 R4, P0, R4, 0x10, RZ ;                 /* 0x0000001004047810 */
                                                                                /* 0x000fea0007f1e0ff */
        /*0090*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x000fe400007fe4ff */
        /*00a0*/                   IADD3 R4, P0, R4, R0, RZ ;                   /* 0x0000000004047210 */
                                                                                /* 0x000fea0007f1e0ff */
        /*00b0*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x000fe400007fe4ff */
        /*00c0*/                   IADD3 R4, P0, R4, 0x194, RZ ;                /* 0x0000019404047810 */
                                                                                /* 0x000fea0007f1e0ff */
        /*00d0*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x000fea00007fe4ff */
        /*00e0*/                   ST.E.NA.U8.STRONG.SYS [R4], RZ ;             /* 0x0000000004007385 */
                                                                                /* 0x000fe200005141ff */
        /*00f0*/                   JMP.U 0x7fac32f75200;                        /* 0x32f752010000794a */
                                                                                /* 0x000fc00003807fac */
        /*0100*/                   BRA 0x100;                                   /* 0xfffffff000007947 */
                                                                                /* 0x000fc0000383ffff */
        /*0110*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0120*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0130*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0140*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0150*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0160*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0170*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0180*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0190*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01a0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01b0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01c0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01d0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01e0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01f0*/                   NOP;                                         /* 0x0000000000007918 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f6c900
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   MOV R2, 0x2 ;                                /* 0x0000000200027802 */
                                                                                /* 0x000fec0000000f00 */
        /*0010*/                   LOP3.LUT P0, RZ, R2, 0x8, RZ, 0xc0, !PT ;    /* 0x0000000802ff7812 */
                                                                                /* 0x000fda000780c0ff */
        /*0020*/               @P0 LOP3.LUT R2, R2, 0x1, RZ, 0xfc, !PT ;        /* 0x0000000102020812 */
                                                                                /* 0x000fe200078efcff */
        /*0030*/                   JMP.U 0x7fac32f5e500;                        /* 0x32f5e5010000794a */
                                                                                /* 0x000fc00003807fac */
        /*0040*/                   BRA 0x40;                                    /* 0xfffffff000007947 */
                                                                                /* 0x000fc0000383ffff */
        /*0050*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0060*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0070*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0080*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0090*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*00a0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*00b0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*00c0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*00d0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*00e0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*00f0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f6d300
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   STL [R1+0x8], R3 ;                           /* 0x0000080301007387 */
                                                                                /* 0x000fe20000100800 */
        /*0010*/                   CCTLL.IV [R1+0x8] ;                          /* 0x0000080001007990 */
                                                                                /* 0x000fe20001800000 */
        /*0020*/                   MOV R4, 0x1e000000 ;                         /* 0x1e00000000047802 */
                                                                                /* 0x000fe40000000f00 */
        /*0030*/                   MOV R5, 0x7fac ;                             /* 0x00007fac00057802 */
                                                                                /* 0x000fe20000000f00 */
        /*0040*/                   S2R R0, SR_VIRTUALSMID ;                     /* 0x0000000000007919 */
                                                                                /* 0x000e220000004300 */
        /*0050*/                   MOV R6, 0x74a20 ;                            /* 0x00074a2000067802 */
                                                                                /* 0x000fec0000000f00 */
        /*0060*/                   IMAD.WIDE.U32 R4, R0, R6, R4 ;               /* 0x0000000600047225 */
                                                                                /* 0x001fea00078e0004 */
        /*0070*/                   IADD3 R4, P0, R4, 0x8, RZ ;                  /* 0x0000000804047810 */
                                                                                /* 0x000fea0007f1e0ff */
        /*0080*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;            /* 0x000000ff05057210 */
                                                                                /* 0x000fe200007fe4ff */
        /*0090*/                   S2R R0, SR_VIRTID ;                          /* 0x0000000000007919 */
                                                                                /* 0x000e240000000300 */
        /*00a0*/                   SHF.R.U32.HI R0, RZ, 0x8, R0 ;               /* 0x00000008ff007819 */
                                                                                /* 0x001fec0000011600 */
        /*00b0*/                   SGXT.U32 R0, R0, 0x7 ;                       /* 0x000000070000781a */
                                                                                /* 0x000fec0000000000 */
        /*00c0*/                   ISETP.GE.U32.AND P0, PT, R0, 0x20, PT ;      /* 0x000000200000780c */
                                                                                /* 0x000fda0003f06070 */
        /*00d0*/               @P0 IADD3 R4, P1, R4, 0x4, RZ ;                  /* 0x0000000404040810 */
                                                                                /* 0x000fea0007f3e0ff */
        /*00e0*/               @P0 IADD3.X R5, R5, RZ, RZ, P1, !PT ;            /* 0x000000ff05050210 */
                                                                                /* 0x000fe40000ffe4ff */
        /*00f0*/               @P0 IADD3 R0, R0, -0x20, RZ ;                    /* 0xffffffe000000810 */
                                                                                /* 0x000fec0007ffe0ff */
        /*0100*/                   SHF.L.U32.HI R0, RZ, R0, 0x1 ;               /* 0x00000001ff007419 */
                                                                                /* 0x000fea0000010600 */
        /*0110*/                   RED.E.OR.STRONG.SM.PRIVATE [R4], R0 ;        /* 0x000000000400798e */
                                                                                /* 0x000fe20003108100 */
        /*0120*/                   RET.ABS R12 0x20;                            /* 0x000000200c007950 */
                                                                                /* 0x000fc00003a00000 */
        /*0130*/                   BRA 0x130;                                   /* 0xfffffff000007947 */
                                                                                /* 0x000fc0000383ffff */
        /*0140*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0150*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0160*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0170*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0180*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*0190*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01a0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01b0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01c0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01d0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01e0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
        /*01f0*/                   NOP;                                         /* 0x0000000000007918 */
                                                                                /* 0x000fc00000000000 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f6de00
                Function Name: do_pause
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   LOP3.LUT P0, RZ, R2, 0x20, RZ, 0xc0, !PT ;         /* 0x0000002002ff7812 */
                                                                                      /* 0x000fe6000780c0ff */
        /*0010*/                   MOV R4, 0x33400000 ;                               /* 0x3340000000047802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0020*/                   MOV R5, 0x7fac ;                                   /* 0x00007fac00057802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0030*/                   MOV R0, 0x1 ;                                      /* 0x0000000100007802 */
                                                                                      /* 0x000fea0000000f00 */
        /*0040*/                   ST.E.STRONG.SM.PRIVATE [R4], R0 ;                  /* 0x0000000004007385 */
                                                                                      /* 0x0007e20000108900 */
        /*0050*/               @P0 BRA.U 0x7c0 ;                                      /* 0x0000076100000947 */
                                                                                      /* 0x000fea0003800000 */
        /*0060*/                   S2R R0, SR_WARPERRORSTATUS ;                       /* 0x0000000000007919 */
                                                                                      /* 0x008e240000004200 */
        /*0070*/                   SHF.R.U32.HI R0, RZ, 0x18, R0 ;                    /* 0x00000018ff007819 */
                                                                                      /* 0x009fec0000011600 */
        /*0080*/                   SGXT.U32 R0, R0, 0x3 ;                             /* 0x000000030000781a */
                                                                                      /* 0x008fec0000000000 */
        /*0090*/                   ISETP.EQ.U32.AND P0, PT, R0, 0x3, PT ;             /* 0x000000030000780c */
                                                                                      /* 0x000fda0003f02070 */
        /*00a0*/              @!P0 BRA.U 0x1e0 ;                                      /* 0x0000013100008947 */
                                                                                      /* 0x000fea0003800000 */
        /*00b0*/                   S2R R0, SR_VIRTID ;                                /* 0x0000000000007919 */
                                                                                      /* 0x008e240000000300 */
        /*00c0*/                   SHF.R.U32.HI R0, RZ, 0x8, R0 ;                     /* 0x00000008ff007819 */
                                                                                      /* 0x009fec0000011600 */
        /*00d0*/                   SGXT.U32 R0, R0, 0x7 ;                             /* 0x000000070000781a */
                                                                                      /* 0x008fe80000000000 */
        /*00e0*/                   S2R R7, SR_VIRTUALSMID ;                           /* 0x0000000000077919 */
                                                                                      /* 0x000e240000004300 */
        /*00f0*/                   IMAD.U32 R2, R7, 0x40, R0 ;                        /* 0x0000004007027824 */
                                                                                      /* 0x001fec00078e0000 */
        /*0100*/                   IMAD.SHL.U32 R2, R2, 0x20, RZ ;                    /* 0x0000002002027824 */
                                                                                      /* 0x000fe400078e00ff */
        /*0110*/                   IMAD.U32 R4, R7, 0x40, R0 ;                        /* 0x0000004007047824 */
                                                                                      /* 0x008fec00078e0000 */
        /*0120*/                   IMAD.SHL.U32 R4, R4, 0x4, RZ ;                     /* 0x0000000404047824 */
                                                                                      /* 0x008fe200078e00ff */
        /*0130*/                   MOV R0, 0x32a2ab00 ;                               /* 0x32a2ab0000007802 */
                                                                                      /* 0x008fe80000000f00 */
        /*0140*/                   MOV R7, 0x7fac ;                                   /* 0x00007fac00077802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0150*/                   IADD3 R2, P0, R0, R2, RZ ;                         /* 0x0000000200027210 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*0160*/                   IADD3.X R3, R7, RZ, RZ, P0, !PT ;                  /* 0x000000ff07037210 */
                                                                                      /* 0x000fe400007fe4ff */
        /*0170*/                   LDC.64 R6, c[0x0][0x1870] ;                        /* 0x00061c00ff067b82 */
                                                                                      /* 0x000ea40000000a00 */
        /*0180*/                   IADD3 R4, P0, R6, R4, RZ ;                         /* 0x0000000406047210 */
                                                                                      /* 0x00cfea0007f1e0ff */
        /*0190*/                   IADD3.X R5, R7, RZ, RZ, P0, !PT ;                  /* 0x000000ff07057210 */
                                                                                      /* 0x008fe200007fe4ff */
        /*01a0*/                   S2R R0, SR_LANEID ;                                /* 0x0000000000007919 */
                                                                                      /* 0x008e240000000000 */
        /*01b0*/                   ISETP.EQ.AND P0, PT, R0, RZ, PT ;                  /* 0x000000ff0000720c */
                                                                                      /* 0x001fda0003f02270 */
        /*01c0*/               @P0 LD.E.CONSTANT R0, [R2+0x4] ;                       /* 0x0000000402000980 */
                                                                                      /* 0x0082240000108900 */
        /*01d0*/               @P0 ST.E.STRONG.SM.PRIVATE [R4], R0 ;                  /* 0x0000000004000385 */
                                                                                      /* 0x0017e40000108900 */
        /*01e0*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x008f220000004300 */
        /*01f0*/                   S2R R5, SR_VIRTID ;                                /* 0x0000000000057919 */
                                                                                      /* 0x008f640000000300 */
        /*0200*/                   SHF.R.U32.HI R5, RZ, 0x8, R5 ;                     /* 0x00000008ff057819 */
                                                                                      /* 0x020fec0000011605 */
        /*0210*/                   SGXT.U32 R5, R5, 0x7 ;                             /* 0x000000070505781a */
                                                                                      /* 0x000fec0000000000 */
        /*0220*/                   IMAD.U32 R0, R4, 0x40, R5 ;                        /* 0x0000004004007824 */
                                                                                      /* 0x019fec00078e0005 */
        /*0230*/                   IMAD R0, R0, 0x198, RZ ;                           /* 0x0000019800007824 */
                                                                                      /* 0x000fe200078e02ff */
        /*0240*/                   MOV R4, 0x33400000 ;                               /* 0x3340000000047802 */
                                                                                      /* 0x000fe80000000f00 */
        /*0250*/                   MOV R5, 0x7fac ;                                   /* 0x00007fac00057802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0260*/                   IADD3 R4, P0, R4, 0x10, RZ ;                       /* 0x0000001004047810 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*0270*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x000fe400007fe4ff */
        /*0280*/                   IADD3 R4, P0, R4, R0, RZ ;                         /* 0x0000000004047210 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*0290*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x000fe400007fe4ff */
        /*02a0*/                   IADD3 R4, P0, R4, 0x195, RZ ;                      /* 0x0000019504047810 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*02b0*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x000fe400007fe4ff */
        /*02c0*/                   MOV R6, 0x1 ;                                      /* 0x0000000100067802 */
                                                                                      /* 0x004fea0000000f00 */
        /*02d0*/                   ST.E.NA.U8.STRONG.SM.PRIVATE [R4], R6 ;            /* 0x0000000004007385 */
                                                                                      /* 0x0001e40000508106 */
        /*02e0*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x001ea20000004300 */
        /*02f0*/                   S2R R5, SR_VIRTID ;                                /* 0x0000000000057919 */
                                                                                      /* 0x001ee40000000300 */
        /*0300*/                   SHF.R.U32.HI R5, RZ, 0x8, R5 ;                     /* 0x00000008ff057819 */
                                                                                      /* 0x009fec0000011605 */
        /*0310*/                   SGXT.U32 R5, R5, 0x7 ;                             /* 0x000000070505781a */
                                                                                      /* 0x001fec0000000000 */
        /*0320*/                   IMAD.U32 R0, R4, 0x40, R5 ;                        /* 0x0000004004007824 */
                                                                                      /* 0x004fec00078e0005 */
        /*0330*/                   IMAD R0, R0, 0x198, RZ ;                           /* 0x0000019800007824 */
                                                                                      /* 0x000fe200078e02ff */
        /*0340*/                   MOV R4, 0x33400000 ;                               /* 0x3340000000047802 */
                                                                                      /* 0x001fe80000000f00 */
        /*0350*/                   MOV R5, 0x7fac ;                                   /* 0x00007fac00057802 */
                                                                                      /* 0x001fe40000000f00 */
        /*0360*/                   IADD3 R4, P0, R4, 0x10, RZ ;                       /* 0x0000001004047810 */
                                                                                      /* 0x001fea0007f1e0ff */
        /*0370*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x001fe400007fe4ff */
        /*0380*/                   IADD3 R4, P0, R4, R0, RZ ;                         /* 0x0000000004047210 */
                                                                                      /* 0x001fea0007f1e0ff */
        /*0390*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x001fe200007fe4ff */
        /*03a0*/                   S2R R6, SR_LANEID ;                                /* 0x0000000000067919 */
                                                                                      /* 0x001ea40000000000 */
        /*03b0*/                   ISETP.EQ.U32.AND P0, PT, R6, RZ, PT ;              /* 0x000000ff0600720c */
                                                                                      /* 0x004fda0003f02070 */
        /*03c0*/               @P0 S2R R0, SR_VIRTID ;                                /* 0x0000000000000919 */
                                                                                      /* 0x000ea40000000300 */
        /*03d0*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x148], R0 ;            /* 0x0000014804000385 */
                                                                                      /* 0x0041e40000108900 */
        /*03e0*/               @P0 S2R R0, SR_CirQueueIncrMinusOne ;                  /* 0x0000000000000919 */
                                                                                      /* 0x001ea40000002900 */
        /*03f0*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x178], R0 ;            /* 0x0000017804000385 */
                                                                                      /* 0x0041e40000108900 */
        /*0400*/               @P0 S2R R0, SR_SMEMSZ ;                                /* 0x0000000000000919 */
                                                                                      /* 0x001ea40000003200 */
        /*0410*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x140], R0 ;            /* 0x0000014004000385 */
                                                                                      /* 0x0041e40000108900 */
        /*0420*/               @P0 S2R R0, SR_LWINSZ ;                                /* 0x0000000000000919 */
                                                                                      /* 0x001ea40000003500 */
        /*0430*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x14c], R0 ;            /* 0x0000014c04000385 */
                                                                                      /* 0x0041e40000108900 */
        /*0440*/               @P0 S2R R0, SR_LMEMLOSZ ;                              /* 0x0000000000000919 */
                                                                                      /* 0x001ea40000003600 */
        /*0450*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x150], R0 ;            /* 0x0000015004000385 */
                                                                                      /* 0x0041e40000108900 */
        /*0460*/               @P0 S2R R0, SR_LMEMHIOFF ;                             /* 0x0000000000000919 */
                                                                                      /* 0x001ea40000003700 */
        /*0470*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x154], R0 ;            /* 0x0000015404000385 */
                                                                                      /* 0x0041e20000108900 */
        /*0480*/                   GETLMEMBASE R2 ;                                   /* 0x00000000000273c0 */
                                                                                      /* 0x002e640000000000 */
        /*0490*/               @P0 ST.E.NA.64.STRONG.SM.PRIVATE [R4+0x158], R2 ;      /* 0x0000015804000385 */
                                                                                      /* 0x0021e40000508b02 */
        /*04a0*/               @P0 S2R R0, SR_GLOBALERRORSTATUS ;                     /* 0x0000000000000919 */
                                                                                      /* 0x001e640000004000 */
        /*04b0*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x144], R0 ;            /* 0x0000014404000385 */
                                                                                      /* 0x0021e40000108900 */
        /*04c0*/               @P0 S2R R0, SR_WARPERRORSTATUS ;                       /* 0x0000000000000919 */
                                                                                      /* 0x001e640000004200 */
        /*04d0*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x164], R0 ;            /* 0x0000016404000385 */
                                                                                      /* 0x0021e40000108900 */
        /*04e0*/                   CS2R R2, SR_ESR_PC ;                               /* 0x0000000000027805 */
                                                                                      /* 0x001fea0000015400 */
        /*04f0*/               @P0 ST.E.NA.64.STRONG.SM.PRIVATE [R4+0x168], R2 ;      /* 0x0000016804000385 */
                                                                                      /* 0x0001e40000508b02 */
        /*0500*/               @P0 S2R R0, SR_CTAID.X ;                               /* 0x0000000000000919 */
                                                                                      /* 0x001e640000002500 */
        /*0510*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x120], R0 ;            /* 0x0000012004000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0520*/               @P0 S2R R0, SR_CTAID.Y ;                               /* 0x0000000000000919 */
                                                                                      /* 0x001e640000002600 */
        /*0530*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x124], R0 ;            /* 0x0000012404000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0540*/               @P0 S2R R0, SR_CTAID.Z ;                               /* 0x0000000000000919 */
                                                                                      /* 0x001e640000002700 */
        /*0550*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x128], R0 ;            /* 0x0000012804000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0560*/               @P0 S2R R0, SR_REGALLOC ;                              /* 0x0000000000000919 */
                                                                                      /* 0x001e640000003d00 */
        /*0570*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x180], R0 ;            /* 0x0000018004000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0580*/                   LDL R2, [0xfffffc] ;                               /* 0xfffffc00ff027983 */
                                                                                      /* 0x001e640000100800 */
        /*0590*/               @P0 ST.E.NA.STRONG.SM.PRIVATE [R4+0x184], R2 ;         /* 0x0000018404000385 */
                                                                                      /* 0x0021e40000508902 */
        /*05a0*/                   LDL.64 R2, [0xffffa8] ;                            /* 0xffffa800ff027983 */
                                                                                      /* 0x001e640000100a00 */
        /*05b0*/               @P0 ST.E.NA.64.STRONG.SM.PRIVATE [R4+0x188], R2 ;      /* 0x0000018804000385 */
                                                                                      /* 0x0021e40000508b02 */
        /*05c0*/                   BMOV.32 R2, MEXITED ;                              /* 0x0000000018027355 */
                                                                                      /* 0x001e640000000000 */
        /*05d0*/               @P0 ST.E.NA.STRONG.SM.PRIVATE [R4+0x190], R2 ;         /* 0x0000019004000385 */
                                                                                      /* 0x0021e40000508902 */
        /*05e0*/               @P0 LDC R0, c[0x0][RZ] ;                               /* 0x00000000ff000b82 */
                                                                                      /* 0x001e640000000800 */
        /*05f0*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x12c], R0 ;            /* 0x0000012c04000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0600*/               @P0 LDC R0, c[0x0][0x4] ;                              /* 0x00000100ff000b82 */
                                                                                      /* 0x001e640000000800 */
        /*0610*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x130], R0 ;            /* 0x0000013004000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0620*/               @P0 LDC R0, c[0x0][0x8] ;                              /* 0x00000200ff000b82 */
                                                                                      /* 0x001e640000000800 */
        /*0630*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x134], R0 ;            /* 0x0000013404000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0640*/               @P0 LDC R0, c[0x0][0xc] ;                              /* 0x00000300ff000b82 */
                                                                                      /* 0x001e640000000800 */
        /*0650*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x10c], R0 ;            /* 0x0000010c04000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0660*/               @P0 LDC R0, c[0x0][0x10] ;                             /* 0x00000400ff000b82 */
                                                                                      /* 0x001e640000000800 */
        /*0670*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x110], R0 ;            /* 0x0000011004000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0680*/               @P0 LDC R0, c[0x0][0x14] ;                             /* 0x00000500ff000b82 */
                                                                                      /* 0x001e640000000800 */
        /*0690*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x114], R0 ;            /* 0x0000011404000385 */
                                                                                      /* 0x0021e40000108900 */
        /*06a0*/               @P0 LDC.64 R2, c[0x0][0x38] ;                          /* 0x00000e00ff020b82 */
                                                                                      /* 0x001e640000000a00 */
        /*06b0*/               @P0 ST.E.64.STRONG.SM.PRIVATE [R4+0x138], R2 ;         /* 0x0000013804000385 */
                                                                                      /* 0x0021e40000108b02 */
        /*06c0*/               @P0 LDC.64 R2, c[0x0][0x40] ;                          /* 0x00001000ff020b82 */
                                                                                      /* 0x001e640000000a00 */
        /*06d0*/               @P0 ST.E.64.STRONG.SM.PRIVATE [R4+0x170], R2 ;         /* 0x0000017004000385 */
                                                                                      /* 0x0021e40000108b02 */
        /*06e0*/               @P0 LDC.64 R2, c[0x0][0x30] ;                          /* 0x00000c00ff020b82 */
                                                                                      /* 0x001e640000000a00 */
        /*06f0*/               @P0 ST.E.64.STRONG.SM.PRIVATE [R4+0x118], R2 ;         /* 0x0000011804000385 */
                                                                                      /* 0x0021e40000108b02 */
        /*0700*/               @P0 B2R.WARP R2 ;                                      /* 0x000000000002031c */
                                                                                      /* 0x001e640000008000 */
        /*0710*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x160], R2 ;            /* 0x0000016004000385 */
                                                                                      /* 0x0021e40000108902 */
        /*0720*/               @P0 LDC R0, c[0x0][0x108] ;                            /* 0x00004200ff000b82 */
                                                                                      /* 0x001e640000000800 */
        /*0730*/               @P0 ST.E.STRONG.SM.PRIVATE [R4+0x100], R0 ;            /* 0x0000010004000385 */
                                                                                      /* 0x0021e40000108900 */
        /*0740*/                   IADD3 R4, P0, R4, 0x0, RZ ;                        /* 0x0000000004047810 */
                                                                                      /* 0x001fea0007f1e0ff */
        /*0750*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x001fec00007fe4ff */
        /*0760*/                   IMAD.WIDE.U32 R4, P0, R6, 0x8, R4 ;                /* 0x0000000806047825 */
                                                                                      /* 0x001fea0007800004 */
        /*0770*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x001fe600007fe4ff */
        /*0780*/                   S2R R0, SR_TID ;                                   /* 0x0000000000007919 */
                                                                                      /* 0x001e640000002000 */
        /*0790*/                   ST.E.STRONG.SM.PRIVATE [R4], R0 ;                  /* 0x0000000004007385 */
                                                                                      /* 0x0021e40000108900 */
        /*07a0*/                   LDL R0, [0xffffc4] ;                               /* 0xffffc400ff007983 */
                                                                                      /* 0x001e240000100800 */
        /*07b0*/                   ST.E.STRONG.SM.PRIVATE [R4+0x4], R0 ;              /* 0x0000000404007385 */
                                                                                      /* 0x001fe20000108900 */
        /*07c0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe40000000000 */
        /*07d0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*07e0*/                   CCTLL.IVALL ;                                      /* 0x00000000ff007990 */
                                                                                      /* 0x000fe20002000000 */
        /*07f0*/                   CCTL.IVALL ;                                       /* 0x00000000ff00798f */
                                                                                      /* 0x000fe20002000000 */
        /*0800*/                   JMP 0x7fac32f79c00 ;                               /* 0x32f79c000000794a */
                                                                                      /* 0x000fea0003807fac */
        /*0810*/                   BPT.DRAIN;                                         /* 0x000000000000795c */
                                                                                      /* 0x000fc00000500000 */
        /*0820*/                   BPT.PAUSE;                                         /* 0x000000000000795c */
                                                                                      /* 0x000fc00000200000 */
        /*0830*/                   CCTL.C.IVALL ;                                     /* 0x00000000ff00798f */
                                                                                      /* 0x000ff00002008000 */
        /*0840*/                   CCTL.I.IVALL ;                                     /* 0x00000000ff00798f */
                                                                                      /* 0x000ff6000200c000 */
        /*0850*/                   CCTLL.IVALL ;                                      /* 0x00000000ff007990 */
                                                                                      /* 0x000fe20002000000 */
        /*0860*/                   CCTL.IVALL ;                                       /* 0x00000000ff00798f */
                                                                                      /* 0x000fe20002000000 */
        /*0870*/                   JMP 0x7fac32f7a600 ;                               /* 0x32f7a6000000794a */
                                                                                      /* 0x000fea0003807fac */
        /*0880*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe40000000000 */
        /*0890*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*08a0*/                   RET.ABS R12 0x20;                                  /* 0x000000200c007950 */
                                                                                      /* 0x000fc00003a00000 */
        /*08b0*/                   BRA 0x8b0;                                         /* 0xfffffff000007947 */
                                                                                      /* 0x000fc0000383ffff */
        /*08c0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*08d0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*08e0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*08f0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0900*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0910*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0920*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0930*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0940*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0950*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0960*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0970*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f6f000
                Function Name: do_preempt_save
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x000e220000004300 */
        /*0010*/                   S2R R5, SR_VIRTID ;                                /* 0x0000000000057919 */
                                                                                      /* 0x000e640000000300 */
        /*0020*/                   SHF.R.U32.HI R5, RZ, 0x8, R5 ;                     /* 0x00000008ff057819 */
                                                                                      /* 0x002fec0000011605 */
        /*0030*/                   SGXT.U32 R5, R5, 0x7 ;                             /* 0x000000070505781a */
                                                                                      /* 0x000fec0000000000 */
        /*0040*/                   IMAD.U32 R0, R4, 0x40, R5 ;                        /* 0x0000004004007824 */
                                                                                      /* 0x001fec00078e0005 */
        /*0050*/                   IMAD R0, R0, 0x198, RZ ;                           /* 0x0000019800007824 */
                                                                                      /* 0x000fe200078e02ff */
        /*0060*/                   MOV R4, 0x33400000 ;                               /* 0x3340000000047802 */
                                                                                      /* 0x000fe80000000f00 */
        /*0070*/                   MOV R5, 0x7fac ;                                   /* 0x00007fac00057802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0080*/                   IADD3 R4, P0, R4, 0x10, RZ ;                       /* 0x0000001004047810 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*0090*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x000fe400007fe4ff */
        /*00a0*/                   IADD3 R4, P0, R4, R0, RZ ;                         /* 0x0000000004047210 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*00b0*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x000fe400007fe4ff */
        /*00c0*/                   IADD3 R4, P0, R4, 0x194, RZ ;                      /* 0x0000019404047810 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*00d0*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x000fe400007fe4ff */
        /*00e0*/                   MOV R0, 0x1 ;                                      /* 0x0000000100007802 */
                                                                                      /* 0x000fea0000000f00 */
        /*00f0*/                   ST.E.NA.U8.STRONG.SYS [R4], R0 ;                   /* 0x0000000004007385 */
                                                                                      /* 0x000fe20000514100 */
        /*0100*/                   JMP.U 0x7fac32f71b00;                              /* 0x32f71b010000794a */
                                                                                      /* 0x000fc00003807fac */
        /*0110*/                   BRA 0x110;                                         /* 0xfffffff000007947 */
                                                                                      /* 0x000fc0000383ffff */
        /*0120*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0130*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0140*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0150*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0160*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0170*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0180*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0190*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01a0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01b0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01c0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01d0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01e0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01f0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
		..........


	code for sm_86
		Entry Point : 0x7fac32f71b00
                Function Name: cilp_traphandler_save
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   S2R R5, SR_LANEID ;                                /* 0x0000000000057919 */
                                                                                      /* 0x000e240000000000 */
        /*0010*/                   ISETP.EQ.AND P0, PT, R5, RZ, PT ;                  /* 0x000000ff0500720c */
                                                                                      /* 0x001fe20003f02270 */
        /*0020*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*0030*/                   B2R R2, 0x0 ;                                      /* 0x000000000002731c */
                                                                                      /* 0x000e220000000000 */
        /*0040*/                   B2R.WARP R3 ;                                      /* 0x000000000003731c */
                                                                                      /* 0x000ea20000008000 */
        /*0050*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*0060*/                   R2B 0x0, RZ ;                                      /* 0x000000ff0000731e */
                                                                                      /* 0x000fe20000000000 */
        /*0070*/                   R2B.WARP 0x0, RZ ;                                 /* 0x000000ff0000731e */
                                                                                      /* 0x000fe20000008000 */
        /*0080*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*0090*/                   BAR.SCAN 0x0, 0x420, PT ;                          /* 0x0010800000007b1d */
                                                                                      /* 0x000fec0003806000 */
        /*00a0*/                   B2R.RESULT R4 ;                                    /* 0x000000000004731c */
                                                                                      /* 0x000e6200000e4000 */
        /*00b0*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*00c0*/                   R2B 0x0, R2 ;                                      /* 0x000000020000731e */
                                                                                      /* 0x0011e20000000000 */
        /*00d0*/                   R2B.WARP 0x0, R3 ;                                 /* 0x000000030000731e */
                                                                                      /* 0x0041e20000008000 */
        /*00e0*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*00f0*/                   ISETP.EQ.AND P1, PT, R4, RZ, PT ;                  /* 0x000000ff0400720c */
                                                                                      /* 0x002fea0003f22270 */
        /*0100*/                   PLOP3.LUT P2, PT, P0, P1, PT, 0x80, 0x8 ;          /* 0x000000000008781c */
                                                                                      /* 0x000fe40000743070 */
        /*0110*/                   MOV R2, 0xf6000000 ;                               /* 0xf600000000027802 */
                                                                                      /* 0x001fe40000000f00 */
        /*0120*/                   MOV R3, 0x7f4a ;                                   /* 0x00007f4a00037802 */
                                                                                      /* 0x001fe20000000f00 */
        /*0130*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x000e620000004300 */
        /*0140*/                   MOV R5, 0x74a20 ;                                  /* 0x00074a2000057802 */
                                                                                      /* 0x000fec0000000f00 */
        /*0150*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x003fea00078e0002 */
        /*0160*/                   IADD3 R2, P3, R2, 0x69020, RZ ;                    /* 0x0006902002027810 */
                                                                                      /* 0x001fea0007f7e0ff */
        /*0170*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x001fe20001ffe4ff */
        /*0180*/                   S2R R4, SR_VIRTID ;                                /* 0x0000000000047919 */
                                                                                      /* 0x000e640000000300 */
        /*0190*/                   SHF.R.U32.HI R4, RZ, 0x8, R4 ;                     /* 0x00000008ff047819 */
                                                                                      /* 0x002fec0000011604 */
        /*01a0*/                   SGXT.U32 R4, R4, 0x7 ;                             /* 0x000000070404781a */
                                                                                      /* 0x000fe40000000000 */
        /*01b0*/                   MOV R5, 0x2c0 ;                                    /* 0x000002c000057802 */
                                                                                      /* 0x000fec0000000f00 */
        /*01c0*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x001fe400078e0002 */
        /*01d0*/                   S2R R4, SR_LANEID ;                                /* 0x0000000000047919 */
                                                                                      /* 0x000e640000000000 */
        /*01e0*/                   ISETP.EQ.AND P3, PT, R4, RZ, PT ;                  /* 0x000000ff0400720c */
                                                                                      /* 0x002fda0003f62270 */
        /*01f0*/               @P3 MOV R4, UR0 ;                                      /* 0x0000000000043c02 */
                                                                                      /* 0x000fe40008000f00 */
        /*0200*/               @P3 MOV R5, UR1 ;                                      /* 0x0000000100053c02 */
                                                                                      /* 0x000fe40008000f00 */
        /*0210*/               @P3 MOV R6, UR2 ;                                      /* 0x0000000200063c02 */
                                                                                      /* 0x000fe40008000f00 */
        /*0220*/               @P3 MOV R7, UR3 ;                                      /* 0x0000000300073c02 */
                                                                                      /* 0x000fea0008000f00 */
        /*0230*/               @P3 ST.E.128 [R2+0x180], R4 ;                          /* 0x0000018002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0240*/               @P3 MOV R4, UR4 ;                                      /* 0x0000000400043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0250*/               @P3 MOV R5, UR5 ;                                      /* 0x0000000500053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0260*/               @P3 MOV R6, UR6 ;                                      /* 0x0000000600063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0270*/               @P3 MOV R7, UR7 ;                                      /* 0x0000000700073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0280*/               @P3 ST.E.128 [R2+0x190], R4 ;                          /* 0x0000019002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0290*/               @P3 MOV R4, UR8 ;                                      /* 0x0000000800043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*02a0*/               @P3 MOV R5, UR9 ;                                      /* 0x0000000900053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*02b0*/               @P3 MOV R6, UR10 ;                                     /* 0x0000000a00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*02c0*/               @P3 MOV R7, UR11 ;                                     /* 0x0000000b00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*02d0*/               @P3 ST.E.128 [R2+0x1a0], R4 ;                          /* 0x000001a002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*02e0*/               @P3 MOV R4, UR12 ;                                     /* 0x0000000c00043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*02f0*/               @P3 MOV R5, UR13 ;                                     /* 0x0000000d00053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0300*/               @P3 MOV R6, UR14 ;                                     /* 0x0000000e00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0310*/               @P3 MOV R7, UR15 ;                                     /* 0x0000000f00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0320*/               @P3 ST.E.128 [R2+0x1b0], R4 ;                          /* 0x000001b002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0330*/               @P3 MOV R4, UR16 ;                                     /* 0x0000001000043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0340*/               @P3 MOV R5, UR17 ;                                     /* 0x0000001100053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0350*/               @P3 MOV R6, UR18 ;                                     /* 0x0000001200063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0360*/               @P3 MOV R7, UR19 ;                                     /* 0x0000001300073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0370*/               @P3 ST.E.128 [R2+0x1c0], R4 ;                          /* 0x000001c002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0380*/               @P3 MOV R4, UR20 ;                                     /* 0x0000001400043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0390*/               @P3 MOV R5, UR21 ;                                     /* 0x0000001500053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*03a0*/               @P3 MOV R6, UR22 ;                                     /* 0x0000001600063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*03b0*/               @P3 MOV R7, UR23 ;                                     /* 0x0000001700073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*03c0*/               @P3 ST.E.128 [R2+0x1d0], R4 ;                          /* 0x000001d002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*03d0*/               @P3 MOV R4, UR24 ;                                     /* 0x0000001800043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*03e0*/               @P3 MOV R5, UR25 ;                                     /* 0x0000001900053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*03f0*/               @P3 MOV R6, UR26 ;                                     /* 0x0000001a00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0400*/               @P3 MOV R7, UR27 ;                                     /* 0x0000001b00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0410*/               @P3 ST.E.128 [R2+0x1e0], R4 ;                          /* 0x000001e002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0420*/               @P3 MOV R4, UR28 ;                                     /* 0x0000001c00043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0430*/               @P3 MOV R5, UR29 ;                                     /* 0x0000001d00053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0440*/               @P3 MOV R6, UR30 ;                                     /* 0x0000001e00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0450*/               @P3 MOV R7, UR31 ;                                     /* 0x0000001f00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0460*/               @P3 ST.E.128 [R2+0x1f0], R4 ;                          /* 0x000001f002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0470*/               @P3 MOV R4, UR32 ;                                     /* 0x0000002000043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0480*/               @P3 MOV R5, UR33 ;                                     /* 0x0000002100053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0490*/               @P3 MOV R6, UR34 ;                                     /* 0x0000002200063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*04a0*/               @P3 MOV R7, UR35 ;                                     /* 0x0000002300073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*04b0*/               @P3 ST.E.128 [R2+0x200], R4 ;                          /* 0x0000020002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*04c0*/               @P3 MOV R4, UR36 ;                                     /* 0x0000002400043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*04d0*/               @P3 MOV R5, UR37 ;                                     /* 0x0000002500053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*04e0*/               @P3 MOV R6, UR38 ;                                     /* 0x0000002600063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*04f0*/               @P3 MOV R7, UR39 ;                                     /* 0x0000002700073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0500*/               @P3 ST.E.128 [R2+0x210], R4 ;                          /* 0x0000021002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0510*/               @P3 MOV R4, UR40 ;                                     /* 0x0000002800043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0520*/               @P3 MOV R5, UR41 ;                                     /* 0x0000002900053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0530*/               @P3 MOV R6, UR42 ;                                     /* 0x0000002a00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0540*/               @P3 MOV R7, UR43 ;                                     /* 0x0000002b00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0550*/               @P3 ST.E.128 [R2+0x220], R4 ;                          /* 0x0000022002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0560*/               @P3 MOV R4, UR44 ;                                     /* 0x0000002c00043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0570*/               @P3 MOV R5, UR45 ;                                     /* 0x0000002d00053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0580*/               @P3 MOV R6, UR46 ;                                     /* 0x0000002e00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0590*/               @P3 MOV R7, UR47 ;                                     /* 0x0000002f00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*05a0*/               @P3 ST.E.128 [R2+0x230], R4 ;                          /* 0x0000023002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*05b0*/               @P3 MOV R4, UR48 ;                                     /* 0x0000003000043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*05c0*/               @P3 MOV R5, UR49 ;                                     /* 0x0000003100053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*05d0*/               @P3 MOV R6, UR50 ;                                     /* 0x0000003200063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*05e0*/               @P3 MOV R7, UR51 ;                                     /* 0x0000003300073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*05f0*/               @P3 ST.E.128 [R2+0x240], R4 ;                          /* 0x0000024002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0600*/               @P3 MOV R4, UR52 ;                                     /* 0x0000003400043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0610*/               @P3 MOV R5, UR53 ;                                     /* 0x0000003500053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0620*/               @P3 MOV R6, UR54 ;                                     /* 0x0000003600063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0630*/               @P3 MOV R7, UR55 ;                                     /* 0x0000003700073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0640*/               @P3 ST.E.128 [R2+0x250], R4 ;                          /* 0x0000025002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*0650*/               @P3 MOV R4, UR56 ;                                     /* 0x0000003800043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0660*/               @P3 MOV R5, UR57 ;                                     /* 0x0000003900053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0670*/               @P3 MOV R6, UR58 ;                                     /* 0x0000003a00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*0680*/               @P3 MOV R7, UR59 ;                                     /* 0x0000003b00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0690*/               @P3 ST.E.128 [R2+0x260], R4 ;                          /* 0x0000026002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*06a0*/               @P3 MOV R4, UR60 ;                                     /* 0x0000003c00043c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*06b0*/               @P3 MOV R5, UR61 ;                                     /* 0x0000003d00053c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*06c0*/               @P3 MOV R6, UR62 ;                                     /* 0x0000003e00063c02 */
                                                                                      /* 0x001fe40008000f00 */
        /*06d0*/               @P3 MOV R7, URZ ;                                      /* 0x0000003f00073c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*06e0*/               @P3 ST.E.128 [R2+0x270], R4 ;                          /* 0x0000027002003385 */
                                                                                      /* 0x0001e40000100d04 */
        /*06f0*/                   S2R R4, SR_LANEID ;                                /* 0x0000000000047919 */
                                                                                      /* 0x001e640000000000 */
        /*0700*/                   ISETP.EQ.AND P4, PT, R4, RZ, PT ;                  /* 0x000000ff0400720c */
                                                                                      /* 0x002fe20003f82270 */
        /*0710*/                   UP2UR UR0, UPR, URZ, 0xff ;                        /* 0x000000ff3f007883 */
                                                                                      /* 0x000fd80008000000 */
        /*0720*/               @P4 MOV R4, UR0 ;                                      /* 0x0000000000044c02 */
                                                                                      /* 0x001fea0008000f00 */
        /*0730*/               @P4 ST.E [R2+0x17c], R4 ;                              /* 0x0000017c02004385 */
                                                                                      /* 0x0001e40000100904 */
        /*0740*/                   S2R R4, SR_TTU_TICKET_INFO ;                       /* 0x0000000000047919 */
                                                                                      /* 0x001e640000008500 */
        /*0750*/                   LOP3.LUT R5, R4, 0x1, RZ, 0xc0, !PT ;              /* 0x0000000104057812 */
                                                                                      /* 0x003fec00078ec0ff */
        /*0760*/                   ISETP.EQ.AND P4, PT, R5, RZ, PT ;                  /* 0x000000ff0500720c */
                                                                                      /* 0x000fe40003f82270 */
        /*0770*/                   S2R R5, SR_LANEID ;                                /* 0x0000000000057919 */
                                                                                      /* 0x001ea40000000000 */
        /*0780*/                   ISETP.EQ.AND P3, PT, R5, RZ, PT ;                  /* 0x000000ff0500720c */
                                                                                      /* 0x004fda0003f62270 */
        /*0790*/               @P3 ST.E [R2+0x280], R4 ;                              /* 0x0000028002003385 */
                                                                                      /* 0x0001e20000100904 */
        /*07a0*/                   BRA.U P4, 0x880 ;                                  /* 0x000000d100007947 */
                                                                                      /* 0x000fea0002000000 */
        /*07b0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*07c0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*07d0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*07e0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*07f0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0800*/                   ISETP.NE.AND P4, PT, RZ, RZ, PT ;                  /* 0x000000ffff00720c */
                                                                                      /* 0x010fe40003f85270 */
                                                                                      /* 0x0003000000ff73d2 */
                                                                                      /* 0x000f2400000804ff */
        /*0820*/                   VOTE.EQ R4, PT, P4 ;                               /* 0x0000000000047806 */
                                                                                      /* 0x013fea00020e0200 */
        /*0830*/               @P3 ST.E [R2+0x284], R4 ;                              /* 0x0000028402003385 */
                                                                                      /* 0x0001e20000100904 */
        /*0840*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0850*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0860*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0870*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0880*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0890*/                   GETLMEMBASE R4 ;                                   /* 0x00000000000473c0 */
                                                                                      /* 0x007e640000000000 */
        /*08a0*/               @P0 ST.E.64 [R2+0x288], R4 ;                           /* 0x0000028802000385 */
                                                                                      /* 0x0021e40000100b04 */
        /*08b0*/                   B2R.WARP R4 ;                                      /* 0x000000000004731c */
                                                                                      /* 0x001e640000008000 */
        /*08c0*/               @P0 ST.E [R2+0x290], R4 ;                              /* 0x0000029002000385 */
                                                                                      /* 0x0021e20000100904 */
        /*08d0*/                   LDL.64 R8, [0xfffff8] ;                            /* 0xfffff800ff087983 */
                                                                                      /* 0x000ea20000100a00 */
        /*08e0*/                   S2R R5, SR_LANEID ;                                /* 0x0000000000057919 */
                                                                                      /* 0x001e640000000000 */
        /*08f0*/                   IADD3 R6, P3, R2, R5, RZ ;                         /* 0x0000000502067210 */
                                                                                      /* 0x003fea0007f7e0ff */
        /*0900*/                   IADD3.X R7, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03077210 */
                                                                                      /* 0x001fea0001ffe4ff */
        /*0910*/                   ST.E.U8 [R6+0x298], R8 ;                           /* 0x0000029806007385 */
                                                                                      /* 0x0041e20000100108 */
        /*0920*/               @P0 ST.E [R2+0x154], R9 ;                              /* 0x0000015402000385 */
                                                                                      /* 0x0001e40000100909 */
        /*0930*/                   LDL.128 R8, [0xffffb0] ;                           /* 0xffffb000ff087983 */
                                                                                      /* 0x001e620000100c00 */
        /*0940*/                   S2R R6, SR_LANEID ;                                /* 0x0000000000067919 */
                                                                                      /* 0x001ea40000000000 */
        /*0950*/                   IMAD.WIDE R6, R6, 0x8, R2 ;                        /* 0x0000000806067825 */
                                                                                      /* 0x005fea00078e0202 */
        /*0960*/                   ST.E.64 [R6], R8 ;                                 /* 0x0000000006007385 */
                                                                                      /* 0x0021e20000100b08 */
        /*0970*/               @P0 ST.E [R2+0x164], R10 ;                             /* 0x0000016402000385 */
                                                                                      /* 0x0001e2000010090a */
        /*0980*/               @P0 ST.E [R2+0x100], R11 ;                             /* 0x0000010002000385 */
                                                                                      /* 0x0001e4000010090b */
        /*0990*/               @P0 LDL.64 R4, [0xffffa8] ;                            /* 0xffffa800ff040983 */
                                                                                      /* 0x001e640000100a00 */
        /*09a0*/               @P0 ST.E.64 [R2+0x140], R4 ;                           /* 0x0000014002000385 */
                                                                                      /* 0x0021e40000100b04 */
        /*09b0*/               @P0 BMOV.32 R4, ATEXIT_PC.HI ;                         /* 0x000000001f040355 */
                                                                                      /* 0x001ea20000000000 */
        /*09c0*/               @P0 BMOV.32 R5, ATEXIT_PC.LO ;                         /* 0x000000001e050355 */
                                                                                      /* 0x001e640000000000 */
        /*09d0*/               @P0 ST.E.64 [R2+0x148], R4 ;                           /* 0x0000014802000385 */
                                                                                      /* 0x0061e40000100b04 */
        /*09e0*/               @P0 BMOV.32 R4, B1 ;                                   /* 0x0000000001040355 */
                                                                                      /* 0x001e640000000000 */
        /*09f0*/               @P0 ST.E [R2+0x104], R4 ;                              /* 0x0000010402000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0a00*/               @P0 BMOV.32 R4, B2 ;                                   /* 0x0000000002040355 */
                                                                                      /* 0x001e640000000000 */
        /*0a10*/               @P0 ST.E [R2+0x108], R4 ;                              /* 0x0000010802000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0a20*/               @P0 BMOV.32 R4, B3 ;                                   /* 0x0000000003040355 */
                                                                                      /* 0x001e640000000000 */
        /*0a30*/               @P0 ST.E [R2+0x10c], R4 ;                              /* 0x0000010c02000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0a40*/               @P0 BMOV.32 R4, B4 ;                                   /* 0x0000000004040355 */
                                                                                      /* 0x001e640000000000 */
        /*0a50*/               @P0 ST.E [R2+0x110], R4 ;                              /* 0x0000011002000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0a60*/               @P0 BMOV.32 R4, B5 ;                                   /* 0x0000000005040355 */
                                                                                      /* 0x001e640000000000 */
        /*0a70*/               @P0 ST.E [R2+0x114], R4 ;                              /* 0x0000011402000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0a80*/               @P0 BMOV.32 R4, B6 ;                                   /* 0x0000000006040355 */
                                                                                      /* 0x001e640000000000 */
        /*0a90*/               @P0 ST.E [R2+0x118], R4 ;                              /* 0x0000011802000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0aa0*/               @P0 BMOV.32 R4, B7 ;                                   /* 0x0000000007040355 */
                                                                                      /* 0x001e640000000000 */
        /*0ab0*/               @P0 ST.E [R2+0x11c], R4 ;                              /* 0x0000011c02000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0ac0*/               @P0 BMOV.32 R4, B8 ;                                   /* 0x0000000008040355 */
                                                                                      /* 0x001e640000000000 */
        /*0ad0*/               @P0 ST.E [R2+0x120], R4 ;                              /* 0x0000012002000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0ae0*/               @P0 BMOV.32 R4, B9 ;                                   /* 0x0000000009040355 */
                                                                                      /* 0x001e640000000000 */
        /*0af0*/               @P0 ST.E [R2+0x124], R4 ;                              /* 0x0000012402000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0b00*/               @P0 BMOV.32 R4, B10 ;                                  /* 0x000000000a040355 */
                                                                                      /* 0x001e640000000000 */
        /*0b10*/               @P0 ST.E [R2+0x128], R4 ;                              /* 0x0000012802000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0b20*/               @P0 BMOV.32 R4, B11 ;                                  /* 0x000000000b040355 */
                                                                                      /* 0x001e640000000000 */
        /*0b30*/               @P0 ST.E [R2+0x12c], R4 ;                              /* 0x0000012c02000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0b40*/               @P0 BMOV.32 R4, B12 ;                                  /* 0x000000000c040355 */
                                                                                      /* 0x001e640000000000 */
        /*0b50*/               @P0 ST.E [R2+0x130], R4 ;                              /* 0x0000013002000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0b60*/               @P0 BMOV.32 R4, B13 ;                                  /* 0x000000000d040355 */
                                                                                      /* 0x001e640000000000 */
        /*0b70*/               @P0 ST.E [R2+0x134], R4 ;                              /* 0x0000013402000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0b80*/               @P0 BMOV.32 R4, B14 ;                                  /* 0x000000000e040355 */
                                                                                      /* 0x001e640000000000 */
        /*0b90*/               @P0 ST.E [R2+0x138], R4 ;                              /* 0x0000013802000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0ba0*/               @P0 BMOV.32 R4, B15 ;                                  /* 0x000000000f040355 */
                                                                                      /* 0x001e640000000000 */
        /*0bb0*/               @P0 ST.E [R2+0x13c], R4 ;                              /* 0x0000013c02000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0bc0*/               @P0 BMOV.32 R4, API_CALL_DEPTH ;                       /* 0x000000001d040355 */
                                                                                      /* 0x001e640000000000 */
        /*0bd0*/               @P0 ST.E [R2+0x160], R4 ;                              /* 0x0000016002000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0be0*/               @P0 BMOV.32 R4, MEXITED ;                              /* 0x0000000018040355 */
                                                                                      /* 0x001e640000000000 */
        /*0bf0*/               @P0 ST.E [R2+0x158], R4 ;                              /* 0x0000015802000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0c00*/               @P0 BMOV.32 R4, MATEXIT ;                              /* 0x000000001b040355 */
                                                                                      /* 0x001e640000000000 */
        /*0c10*/               @P0 ST.E [R2+0x15c], R4 ;                              /* 0x0000015c02000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0c20*/               @P0 BMOV.32 R4, THREAD_STATE_ENUM.0 ;                  /* 0x0000000010040355 */
                                                                                      /* 0x001e640000000000 */
        /*0c30*/               @P0 ST.E [R2+0x168], R4 ;                              /* 0x0000016802000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0c40*/               @P0 BMOV.32 R4, THREAD_STATE_ENUM.1 ;                  /* 0x0000000011040355 */
                                                                                      /* 0x001e640000000000 */
        /*0c50*/               @P0 ST.E [R2+0x16c], R4 ;                              /* 0x0000016c02000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0c60*/               @P0 BMOV.32 R4, THREAD_STATE_ENUM.2 ;                  /* 0x0000000012040355 */
                                                                                      /* 0x001e640000000000 */
        /*0c70*/               @P0 ST.E [R2+0x170], R4 ;                              /* 0x0000017002000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0c80*/               @P0 BMOV.32 R4, THREAD_STATE_ENUM.3 ;                  /* 0x0000000013040355 */
                                                                                      /* 0x001e640000000000 */
        /*0c90*/               @P0 ST.E [R2+0x174], R4 ;                              /* 0x0000017402000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0ca0*/               @P0 BMOV.32 R4, THREAD_STATE_ENUM.4 ;                  /* 0x0000000014040355 */
                                                                                      /* 0x001e640000000000 */
        /*0cb0*/               @P0 ST.E [R2+0x178], R4 ;                              /* 0x0000017802000385 */
                                                                                      /* 0x0021e40000100904 */
        /*0cc0*/                   MOV R2, 0xf6000000 ;                               /* 0xf600000000027802 */
                                                                                      /* 0x001fe40000000f00 */
        /*0cd0*/                   MOV R3, 0x7f4a ;                                   /* 0x00007f4a00037802 */
                                                                                      /* 0x001fe20000000f00 */
        /*0ce0*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x001e620000004300 */
        /*0cf0*/                   MOV R5, 0x74a20 ;                                  /* 0x00074a2000057802 */
                                                                                      /* 0x001fec0000000f00 */
        /*0d00*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x003fe400078e0002 */
        /*0d10*/                   S2R R4, SR_REGALLOC ;                              /* 0x0000000000047919 */
                                                                                      /* 0x001e640000003d00 */
        /*0d20*/                   IADD3 R4, R4, -0x2, RZ ;                           /* 0xfffffffe04047810 */
                                                                                      /* 0x003fec0007ffe0ff */
        /*0d30*/                   SHF.L.U32 R4, R4, 0x7, RZ ;                        /* 0x0000000704047819 */
                                                                                      /* 0x001fea00000006ff */
        /*0d40*/               @P0 ATOM.E.ADD PT, R0, [R2], R4 ;                      /* 0x000000040200038a */
                                                                                      /* 0x00006400001e0100 */
        /*0d50*/                   SHFL.IDX PT, R0, R0, 0x0, 0x0 ;                    /* 0x0000000000007f89 */
                                                                                      /* 0x0022e400000e0000 */
        /*0d60*/                   IADD3 R2, P3, R2, R0, RZ ;                         /* 0x0000000002027210 */
                                                                                      /* 0x009fea0007f7e0ff */
        /*0d70*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x001fe20001ffe4ff */
        /*0d80*/                   S2R R5, SR_LANEID ;                                /* 0x0000000000057919 */
                                                                                      /* 0x001e640000000000 */
        /*0d90*/                   SHF.L.U32 R5, R5, 0x4, RZ ;                        /* 0x0000000405057819 */
                                                                                      /* 0x003fec00000006ff */
        /*0da0*/                   IADD3 R2, P3, R2, R5, RZ ;                         /* 0x0000000502027210 */
                                                                                      /* 0x001fea0007f7e0ff */
        /*0db0*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x001fe60001ffe4ff */
        /*0dc0*/                   LDL.128 R4, [0xffffc0] ;                           /* 0xffffc000ff047983 */
                                                                                      /* 0x001e640000100c00 */
        /*0dd0*/                   ST.E.128 [R2+0x20], R4 ;                           /* 0x0000002002007385 */
                                                                                      /* 0x0021e40000100d04 */
        /*0de0*/                   LDL.128 R4, [0xffffd0] ;                           /* 0xffffd000ff047983 */
                                                                                      /* 0x001e640000100c00 */
        /*0df0*/                   ST.E.128 [R2+0x220], R4 ;                          /* 0x0000022002007385 */
                                                                                      /* 0x0021e40000100d04 */
        /*0e00*/                   LDL.128 R4, [0xffffe0] ;                           /* 0xffffe000ff047983 */
                                                                                      /* 0x001e640000100c00 */
        /*0e10*/                   ST.E.128 [R2+0x420], R4 ;                          /* 0x0000042002007385 */
                                                                                      /* 0x0021e40000100d04 */
        /*0e20*/                   LDL.64 R8, [0xfffff0] ;                            /* 0xfffff000ff087983 */
                                                                                      /* 0x001f620000100a00 */
        /*0e30*/                   S2R R4, SR_REGALLOC ;                              /* 0x0000000000047919 */
                                                                                      /* 0x001ea40000003d00 */
        /*0e40*/                   IADD3 R4, R4, -0x2, RZ ;                           /* 0xfffffffe04047810 */
                                                                                      /* 0x005fec0007ffe0ff */
        /*0e50*/                   ISETP.GE.AND P3, PT, R4, 0x10, PT ;                /* 0x000000100400780c */
                                                                                      /* 0x000fe40003f66270 */
        /*0e60*/                   ISETP.GE.AND P4, PT, R4, 0xf, PT ;                 /* 0x0000000f0400780c */
                                                                                      /* 0x010ff60003f86270 */
        /*0e70*/               @P3 BRA.U 0xea0 ;                                      /* 0x0000002100003947 */
                                                                                      /* 0x000fea0003800000 */
        /*0e80*/               @P4 BRA.U 0xeb0 ;                                      /* 0x0000002100004947 */
                                                                                      /* 0x000fea0003800000 */
        /*0e90*/                   BRA.U 0xec0 ;                                      /* 0x0000002100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*0ea0*/                   MOV R11, R15 ;                                     /* 0x0000000f000b7202 */
                                                                                      /* 0x003fe20000000f00 */
        /*0eb0*/                   MOV R10, R14 ;                                     /* 0x0000000e000a7202 */
                                                                                      /* 0x003fe20000000f00 */
        /*0ec0*/                   MOV R7, RZ ;                                       /* 0x000000ff00077202 */
                                                                                      /* 0x003fe40000000f00 */
        /*0ed0*/                   LOP3.LUT R5, R4, 0xfc, RZ, 0xc0, !PT ;             /* 0x000000fc04057812 */
                                                                                      /* 0x003fec00078ec0ff */
        /*0ee0*/                   IADD3 R5, -R5, 0xfc, RZ ;                          /* 0x000000fc05057810 */
                                                                                      /* 0x000fec0007ffe1ff */
        /*0ef0*/                   SHF.L.U32 R6, R5, 0x2, RZ ;                        /* 0x0000000205067819 */
                                                                                      /* 0x003fe400000006ff */
        /*0f00*/                   ISETP.LT.AND P3, PT, R4, 0x10, PT ;                /* 0x000000100400780c */
                                                                                      /* 0x000fda0003f61270 */
        /*0f10*/               @P3 BRA.U 0x1310 ;                                     /* 0x000003f100003947 */
                                                                                      /* 0x000fea0003800000 */
        /*0f20*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0f30*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*0f40*/                   BRX R6 ;                                           /* 0x0000000006007949 */
                                                                                      /* 0x000fe20003800000 */
        /*0f50*/                   ST.E.128 [R2+0x7c20], R248 ;                       /* 0x00007c2002007385 */
                                                                                      /* 0x0001e20000100df8 */
        /*0f60*/                   ST.E.128 [R2+0x7a20], R244 ;                       /* 0x00007a2002007385 */
                                                                                      /* 0x0001e20000100df4 */
        /*0f70*/                   ST.E.128 [R2+0x7820], R240 ;                       /* 0x0000782002007385 */
                                                                                      /* 0x0001e20000100df0 */
        /*0f80*/                   ST.E.128 [R2+0x7620], R236 ;                       /* 0x0000762002007385 */
                                                                                      /* 0x0001e20000100dec */
        /*0f90*/                   ST.E.128 [R2+0x7420], R232 ;                       /* 0x0000742002007385 */
                                                                                      /* 0x0001e20000100de8 */
        /*0fa0*/                   ST.E.128 [R2+0x7220], R228 ;                       /* 0x0000722002007385 */
                                                                                      /* 0x0001e20000100de4 */
        /*0fb0*/                   ST.E.128 [R2+0x7020], R224 ;                       /* 0x0000702002007385 */
                                                                                      /* 0x0001e20000100de0 */
        /*0fc0*/                   ST.E.128 [R2+0x6e20], R220 ;                       /* 0x00006e2002007385 */
                                                                                      /* 0x0001e20000100ddc */
        /*0fd0*/                   ST.E.128 [R2+0x6c20], R216 ;                       /* 0x00006c2002007385 */
                                                                                      /* 0x0001e20000100dd8 */
        /*0fe0*/                   ST.E.128 [R2+0x6a20], R212 ;                       /* 0x00006a2002007385 */
                                                                                      /* 0x0001e20000100dd4 */
        /*0ff0*/                   ST.E.128 [R2+0x6820], R208 ;                       /* 0x0000682002007385 */
                                                                                      /* 0x0001e20000100dd0 */
        /*1000*/                   ST.E.128 [R2+0x6620], R204 ;                       /* 0x0000662002007385 */
                                                                                      /* 0x0001e20000100dcc */
        /*1010*/                   ST.E.128 [R2+0x6420], R200 ;                       /* 0x0000642002007385 */
                                                                                      /* 0x0001e20000100dc8 */
        /*1020*/                   ST.E.128 [R2+0x6220], R196 ;                       /* 0x0000622002007385 */
                                                                                      /* 0x0001e20000100dc4 */
        /*1030*/                   ST.E.128 [R2+0x6020], R192 ;                       /* 0x0000602002007385 */
                                                                                      /* 0x0001e20000100dc0 */
        /*1040*/                   ST.E.128 [R2+0x5e20], R188 ;                       /* 0x00005e2002007385 */
                                                                                      /* 0x0001e20000100dbc */
        /*1050*/                   ST.E.128 [R2+0x5c20], R184 ;                       /* 0x00005c2002007385 */
                                                                                      /* 0x0001e20000100db8 */
        /*1060*/                   ST.E.128 [R2+0x5a20], R180 ;                       /* 0x00005a2002007385 */
                                                                                      /* 0x0001e20000100db4 */
        /*1070*/                   ST.E.128 [R2+0x5820], R176 ;                       /* 0x0000582002007385 */
                                                                                      /* 0x0001e20000100db0 */
        /*1080*/                   ST.E.128 [R2+0x5620], R172 ;                       /* 0x0000562002007385 */
                                                                                      /* 0x0001e20000100dac */
        /*1090*/                   ST.E.128 [R2+0x5420], R168 ;                       /* 0x0000542002007385 */
                                                                                      /* 0x0001e20000100da8 */
        /*10a0*/                   ST.E.128 [R2+0x5220], R164 ;                       /* 0x0000522002007385 */
                                                                                      /* 0x0001e20000100da4 */
        /*10b0*/                   ST.E.128 [R2+0x5020], R160 ;                       /* 0x0000502002007385 */
                                                                                      /* 0x0001e20000100da0 */
        /*10c0*/                   ST.E.128 [R2+0x4e20], R156 ;                       /* 0x00004e2002007385 */
                                                                                      /* 0x0001e20000100d9c */
        /*10d0*/                   ST.E.128 [R2+0x4c20], R152 ;                       /* 0x00004c2002007385 */
                                                                                      /* 0x0001e20000100d98 */
        /*10e0*/                   ST.E.128 [R2+0x4a20], R148 ;                       /* 0x00004a2002007385 */
                                                                                      /* 0x0001e20000100d94 */
        /*10f0*/                   ST.E.128 [R2+0x4820], R144 ;                       /* 0x0000482002007385 */
                                                                                      /* 0x0001e20000100d90 */
        /*1100*/                   ST.E.128 [R2+0x4620], R140 ;                       /* 0x0000462002007385 */
                                                                                      /* 0x0001e20000100d8c */
        /*1110*/                   ST.E.128 [R2+0x4420], R136 ;                       /* 0x0000442002007385 */
                                                                                      /* 0x0001e20000100d88 */
        /*1120*/                   ST.E.128 [R2+0x4220], R132 ;                       /* 0x0000422002007385 */
                                                                                      /* 0x0001e20000100d84 */
        /*1130*/                   ST.E.128 [R2+0x4020], R128 ;                       /* 0x0000402002007385 */
                                                                                      /* 0x0001e20000100d80 */
        /*1140*/                   ST.E.128 [R2+0x3e20], R124 ;                       /* 0x00003e2002007385 */
                                                                                      /* 0x0001e20000100d7c */
        /*1150*/                   ST.E.128 [R2+0x3c20], R120 ;                       /* 0x00003c2002007385 */
                                                                                      /* 0x0001e20000100d78 */
        /*1160*/                   ST.E.128 [R2+0x3a20], R116 ;                       /* 0x00003a2002007385 */
                                                                                      /* 0x0001e20000100d74 */
        /*1170*/                   ST.E.128 [R2+0x3820], R112 ;                       /* 0x0000382002007385 */
                                                                                      /* 0x0001e20000100d70 */
        /*1180*/                   ST.E.128 [R2+0x3620], R108 ;                       /* 0x0000362002007385 */
                                                                                      /* 0x0001e20000100d6c */
        /*1190*/                   ST.E.128 [R2+0x3420], R104 ;                       /* 0x0000342002007385 */
                                                                                      /* 0x0001e20000100d68 */
        /*11a0*/                   ST.E.128 [R2+0x3220], R100 ;                       /* 0x0000322002007385 */
                                                                                      /* 0x0001e20000100d64 */
        /*11b0*/                   ST.E.128 [R2+0x3020], R96 ;                        /* 0x0000302002007385 */
                                                                                      /* 0x0001e20000100d60 */
        /*11c0*/                   ST.E.128 [R2+0x2e20], R92 ;                        /* 0x00002e2002007385 */
                                                                                      /* 0x0001e20000100d5c */
        /*11d0*/                   ST.E.128 [R2+0x2c20], R88 ;                        /* 0x00002c2002007385 */
                                                                                      /* 0x0001e20000100d58 */
        /*11e0*/                   ST.E.128 [R2+0x2a20], R84 ;                        /* 0x00002a2002007385 */
                                                                                      /* 0x0001e20000100d54 */
        /*11f0*/                   ST.E.128 [R2+0x2820], R80 ;                        /* 0x0000282002007385 */
                                                                                      /* 0x0001e20000100d50 */
        /*1200*/                   ST.E.128 [R2+0x2620], R76 ;                        /* 0x0000262002007385 */
                                                                                      /* 0x0001e20000100d4c */
        /*1210*/                   ST.E.128 [R2+0x2420], R72 ;                        /* 0x0000242002007385 */
                                                                                      /* 0x0001e20000100d48 */
        /*1220*/                   ST.E.128 [R2+0x2220], R68 ;                        /* 0x0000222002007385 */
                                                                                      /* 0x0001e20000100d44 */
        /*1230*/                   ST.E.128 [R2+0x2020], R64 ;                        /* 0x0000202002007385 */
                                                                                      /* 0x0001e20000100d40 */
        /*1240*/                   ST.E.128 [R2+0x1e20], R60 ;                        /* 0x00001e2002007385 */
                                                                                      /* 0x0001e20000100d3c */
        /*1250*/                   ST.E.128 [R2+0x1c20], R56 ;                        /* 0x00001c2002007385 */
                                                                                      /* 0x0001e20000100d38 */
        /*1260*/                   ST.E.128 [R2+0x1a20], R52 ;                        /* 0x00001a2002007385 */
                                                                                      /* 0x0001e20000100d34 */
        /*1270*/                   ST.E.128 [R2+0x1820], R48 ;                        /* 0x0000182002007385 */
                                                                                      /* 0x0001e20000100d30 */
        /*1280*/                   ST.E.128 [R2+0x1620], R44 ;                        /* 0x0000162002007385 */
                                                                                      /* 0x0001e20000100d2c */
        /*1290*/                   ST.E.128 [R2+0x1420], R40 ;                        /* 0x0000142002007385 */
                                                                                      /* 0x0001e20000100d28 */
        /*12a0*/                   ST.E.128 [R2+0x1220], R36 ;                        /* 0x0000122002007385 */
                                                                                      /* 0x0001e20000100d24 */
        /*12b0*/                   ST.E.128 [R2+0x1020], R32 ;                        /* 0x0000102002007385 */
                                                                                      /* 0x0001e20000100d20 */
        /*12c0*/                   ST.E.128 [R2+0xe20], R28 ;                         /* 0x00000e2002007385 */
                                                                                      /* 0x0001e20000100d1c */
        /*12d0*/                   ST.E.128 [R2+0xc20], R24 ;                         /* 0x00000c2002007385 */
                                                                                      /* 0x0001e20000100d18 */
        /*12e0*/                   ST.E.128 [R2+0xa20], R20 ;                         /* 0x00000a2002007385 */
                                                                                      /* 0x0001e20000100d14 */
        /*12f0*/                   ST.E.128 [R2+0x820], R16 ;                         /* 0x0000082002007385 */
                                                                                      /* 0x0001e20000100d10 */
        /*1300*/                   ST.E.128 [R2+0x620], R8 ;                          /* 0x0000062002007385 */
                                                                                      /* 0x0221e20000100d08 */
        /*1310*/                   S2R R6, SR_LANEID ;                                /* 0x0000000000067919 */
                                                                                      /* 0x000ea40000000000 */
        /*1320*/                   SHF.L.U32 R5, R6, 0x4, RZ ;                        /* 0x0000000406057819 */
                                                                                      /* 0x004fec00000006ff */
        /*1330*/                   IADD3 R2, P3, R2, -R5, RZ ;                        /* 0x8000000502027210 */
                                                                                      /* 0x001fea0007f7e0ff */
        /*1340*/                   IADD3.X R3, R3, ~RZ, RZ, P3, !PT ;                 /* 0x800000ff03037210 */
                                                                                      /* 0x001fe40001ffe4ff */
        /*1350*/                   SHF.L.U32 R5, R6, 0x2, RZ ;                        /* 0x0000000206057819 */
                                                                                      /* 0x000fec00000006ff */
        /*1360*/                   IADD3 R2, P3, R2, R5, RZ ;                         /* 0x0000000502027210 */
                                                                                      /* 0x000fea0007f7e0ff */
        /*1370*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x000fe40001ffe4ff */
        /*1380*/                   LOP3.LUT R5, R4, 0x3, RZ, 0xc0, !PT ;              /* 0x0000000304057812 */
                                                                                      /* 0x000fec00078ec0ff */
        /*1390*/                   ISETP.EQ.AND P4, PT, R5, 0x0, PT ;                 /* 0x000000000500780c */
                                                                                      /* 0x000fda0003f82270 */
        /*13a0*/               @P4 BRA.U 0x2350 ;                                     /* 0x00000fa100004947 */
                                                                                      /* 0x000fea0003800000 */
        /*13b0*/                   MOV R7, RZ ;                                       /* 0x000000ff00077202 */
                                                                                      /* 0x000fe40000000f00 */
        /*13c0*/                   IADD3 R5, -R4, 0x100, RZ ;                         /* 0x0000010004057810 */
                                                                                      /* 0x000fec0007ffe1ff */
        /*13d0*/                   LEA R6, R5, 0xfffffff0, 0x4 ;                      /* 0xfffffff005067811 */
                                                                                      /* 0x000fe800078e20ff */
        /*13e0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*13f0*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*1400*/                   BRX R6 ;                                           /* 0x0000000006007949 */
                                                                                      /* 0x004fe20003800000 */
        /*1410*/                   NOP ;                                              /* 0x0000000000007918 */
                                                                                      /* 0x000fe20000000000 */
        /*1420*/                   ST.E [R2+0x7ea0], R253 ;                           /* 0x00007ea002007385 */
                                                                                      /* 0x0001e200001009fd */
        /*1430*/                   ST.E [R2+0x7e20], R252 ;                           /* 0x00007e2002007385 */
                                                                                      /* 0x0001e200001009fc */
        /*1440*/                   BRA.U 0x2350 ;                                     /* 0x00000f0100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1450*/                   ST.E [R2+0x7d20], R250 ;                           /* 0x00007d2002007385 */
                                                                                      /* 0x0001e200001009fa */
        /*1460*/                   ST.E [R2+0x7ca0], R249 ;                           /* 0x00007ca002007385 */
                                                                                      /* 0x0001e200001009f9 */
        /*1470*/                   ST.E [R2+0x7c20], R248 ;                           /* 0x00007c2002007385 */
                                                                                      /* 0x0001e200001009f8 */
        /*1480*/                   BRA.U 0x2350 ;                                     /* 0x00000ec100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1490*/                   ST.E [R2+0x7b20], R246 ;                           /* 0x00007b2002007385 */
                                                                                      /* 0x0001e200001009f6 */
        /*14a0*/                   ST.E [R2+0x7aa0], R245 ;                           /* 0x00007aa002007385 */
                                                                                      /* 0x0001e200001009f5 */
        /*14b0*/                   ST.E [R2+0x7a20], R244 ;                           /* 0x00007a2002007385 */
                                                                                      /* 0x0001e200001009f4 */
        /*14c0*/                   BRA.U 0x2350 ;                                     /* 0x00000e8100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*14d0*/                   ST.E [R2+0x7920], R242 ;                           /* 0x0000792002007385 */
                                                                                      /* 0x0001e200001009f2 */
        /*14e0*/                   ST.E [R2+0x78a0], R241 ;                           /* 0x000078a002007385 */
                                                                                      /* 0x0001e200001009f1 */
        /*14f0*/                   ST.E [R2+0x7820], R240 ;                           /* 0x0000782002007385 */
                                                                                      /* 0x0001e200001009f0 */
        /*1500*/                   BRA.U 0x2350 ;                                     /* 0x00000e4100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1510*/                   ST.E [R2+0x7720], R238 ;                           /* 0x0000772002007385 */
                                                                                      /* 0x0001e200001009ee */
        /*1520*/                   ST.E [R2+0x76a0], R237 ;                           /* 0x000076a002007385 */
                                                                                      /* 0x0001e200001009ed */
        /*1530*/                   ST.E [R2+0x7620], R236 ;                           /* 0x0000762002007385 */
                                                                                      /* 0x0001e200001009ec */
        /*1540*/                   BRA.U 0x2350 ;                                     /* 0x00000e0100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1550*/                   ST.E [R2+0x7520], R234 ;                           /* 0x0000752002007385 */
                                                                                      /* 0x0001e200001009ea */
        /*1560*/                   ST.E [R2+0x74a0], R233 ;                           /* 0x000074a002007385 */
                                                                                      /* 0x0001e200001009e9 */
        /*1570*/                   ST.E [R2+0x7420], R232 ;                           /* 0x0000742002007385 */
                                                                                      /* 0x0001e200001009e8 */
        /*1580*/                   BRA.U 0x2350 ;                                     /* 0x00000dc100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1590*/                   ST.E [R2+0x7320], R230 ;                           /* 0x0000732002007385 */
                                                                                      /* 0x0001e200001009e6 */
        /*15a0*/                   ST.E [R2+0x72a0], R229 ;                           /* 0x000072a002007385 */
                                                                                      /* 0x0001e200001009e5 */
        /*15b0*/                   ST.E [R2+0x7220], R228 ;                           /* 0x0000722002007385 */
                                                                                      /* 0x0001e200001009e4 */
        /*15c0*/                   BRA.U 0x2350 ;                                     /* 0x00000d8100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*15d0*/                   ST.E [R2+0x7120], R226 ;                           /* 0x0000712002007385 */
                                                                                      /* 0x0001e200001009e2 */
        /*15e0*/                   ST.E [R2+0x70a0], R225 ;                           /* 0x000070a002007385 */
                                                                                      /* 0x0001e200001009e1 */
        /*15f0*/                   ST.E [R2+0x7020], R224 ;                           /* 0x0000702002007385 */
                                                                                      /* 0x0001e200001009e0 */
        /*1600*/                   BRA.U 0x2350 ;                                     /* 0x00000d4100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1610*/                   ST.E [R2+0x6f20], R222 ;                           /* 0x00006f2002007385 */
                                                                                      /* 0x0001e200001009de */
        /*1620*/                   ST.E [R2+0x6ea0], R221 ;                           /* 0x00006ea002007385 */
                                                                                      /* 0x0001e200001009dd */
        /*1630*/                   ST.E [R2+0x6e20], R220 ;                           /* 0x00006e2002007385 */
                                                                                      /* 0x0001e200001009dc */
        /*1640*/                   BRA.U 0x2350 ;                                     /* 0x00000d0100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1650*/                   ST.E [R2+0x6d20], R218 ;                           /* 0x00006d2002007385 */
                                                                                      /* 0x0001e200001009da */
        /*1660*/                   ST.E [R2+0x6ca0], R217 ;                           /* 0x00006ca002007385 */
                                                                                      /* 0x0001e200001009d9 */
        /*1670*/                   ST.E [R2+0x6c20], R216 ;                           /* 0x00006c2002007385 */
                                                                                      /* 0x0001e200001009d8 */
        /*1680*/                   BRA.U 0x2350 ;                                     /* 0x00000cc100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1690*/                   ST.E [R2+0x6b20], R214 ;                           /* 0x00006b2002007385 */
                                                                                      /* 0x0001e200001009d6 */
        /*16a0*/                   ST.E [R2+0x6aa0], R213 ;                           /* 0x00006aa002007385 */
                                                                                      /* 0x0001e200001009d5 */
        /*16b0*/                   ST.E [R2+0x6a20], R212 ;                           /* 0x00006a2002007385 */
                                                                                      /* 0x0001e200001009d4 */
        /*16c0*/                   BRA.U 0x2350 ;                                     /* 0x00000c8100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*16d0*/                   ST.E [R2+0x6920], R210 ;                           /* 0x0000692002007385 */
                                                                                      /* 0x0001e200001009d2 */
        /*16e0*/                   ST.E [R2+0x68a0], R209 ;                           /* 0x000068a002007385 */
                                                                                      /* 0x0001e200001009d1 */
        /*16f0*/                   ST.E [R2+0x6820], R208 ;                           /* 0x0000682002007385 */
                                                                                      /* 0x0001e200001009d0 */
        /*1700*/                   BRA.U 0x2350 ;                                     /* 0x00000c4100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1710*/                   ST.E [R2+0x6720], R206 ;                           /* 0x0000672002007385 */
                                                                                      /* 0x0001e200001009ce */
        /*1720*/                   ST.E [R2+0x66a0], R205 ;                           /* 0x000066a002007385 */
                                                                                      /* 0x0001e200001009cd */
        /*1730*/                   ST.E [R2+0x6620], R204 ;                           /* 0x0000662002007385 */
                                                                                      /* 0x0001e200001009cc */
        /*1740*/                   BRA.U 0x2350 ;                                     /* 0x00000c0100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1750*/                   ST.E [R2+0x6520], R202 ;                           /* 0x0000652002007385 */
                                                                                      /* 0x0001e200001009ca */
        /*1760*/                   ST.E [R2+0x64a0], R201 ;                           /* 0x000064a002007385 */
                                                                                      /* 0x0001e200001009c9 */
        /*1770*/                   ST.E [R2+0x6420], R200 ;                           /* 0x0000642002007385 */
                                                                                      /* 0x0001e200001009c8 */
        /*1780*/                   BRA.U 0x2350 ;                                     /* 0x00000bc100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1790*/                   ST.E [R2+0x6320], R198 ;                           /* 0x0000632002007385 */
                                                                                      /* 0x0001e200001009c6 */
        /*17a0*/                   ST.E [R2+0x62a0], R197 ;                           /* 0x000062a002007385 */
                                                                                      /* 0x0001e200001009c5 */
        /*17b0*/                   ST.E [R2+0x6220], R196 ;                           /* 0x0000622002007385 */
                                                                                      /* 0x0001e200001009c4 */
        /*17c0*/                   BRA.U 0x2350 ;                                     /* 0x00000b8100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*17d0*/                   ST.E [R2+0x6120], R194 ;                           /* 0x0000612002007385 */
                                                                                      /* 0x0001e200001009c2 */
        /*17e0*/                   ST.E [R2+0x60a0], R193 ;                           /* 0x000060a002007385 */
                                                                                      /* 0x0001e200001009c1 */
        /*17f0*/                   ST.E [R2+0x6020], R192 ;                           /* 0x0000602002007385 */
                                                                                      /* 0x0001e200001009c0 */
        /*1800*/                   BRA.U 0x2350 ;                                     /* 0x00000b4100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1810*/                   ST.E [R2+0x5f20], R190 ;                           /* 0x00005f2002007385 */
                                                                                      /* 0x0001e200001009be */
        /*1820*/                   ST.E [R2+0x5ea0], R189 ;                           /* 0x00005ea002007385 */
                                                                                      /* 0x0001e200001009bd */
        /*1830*/                   ST.E [R2+0x5e20], R188 ;                           /* 0x00005e2002007385 */
                                                                                      /* 0x0001e200001009bc */
        /*1840*/                   BRA.U 0x2350 ;                                     /* 0x00000b0100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1850*/                   ST.E [R2+0x5d20], R186 ;                           /* 0x00005d2002007385 */
                                                                                      /* 0x0001e200001009ba */
        /*1860*/                   ST.E [R2+0x5ca0], R185 ;                           /* 0x00005ca002007385 */
                                                                                      /* 0x0001e200001009b9 */
        /*1870*/                   ST.E [R2+0x5c20], R184 ;                           /* 0x00005c2002007385 */
                                                                                      /* 0x0001e200001009b8 */
        /*1880*/                   BRA.U 0x2350 ;                                     /* 0x00000ac100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1890*/                   ST.E [R2+0x5b20], R182 ;                           /* 0x00005b2002007385 */
                                                                                      /* 0x0001e200001009b6 */
        /*18a0*/                   ST.E [R2+0x5aa0], R181 ;                           /* 0x00005aa002007385 */
                                                                                      /* 0x0001e200001009b5 */
        /*18b0*/                   ST.E [R2+0x5a20], R180 ;                           /* 0x00005a2002007385 */
                                                                                      /* 0x0001e200001009b4 */
        /*18c0*/                   BRA.U 0x2350 ;                                     /* 0x00000a8100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*18d0*/                   ST.E [R2+0x5920], R178 ;                           /* 0x0000592002007385 */
                                                                                      /* 0x0001e200001009b2 */
        /*18e0*/                   ST.E [R2+0x58a0], R177 ;                           /* 0x000058a002007385 */
                                                                                      /* 0x0001e200001009b1 */
        /*18f0*/                   ST.E [R2+0x5820], R176 ;                           /* 0x0000582002007385 */
                                                                                      /* 0x0001e200001009b0 */
        /*1900*/                   BRA.U 0x2350 ;                                     /* 0x00000a4100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1910*/                   ST.E [R2+0x5720], R174 ;                           /* 0x0000572002007385 */
                                                                                      /* 0x0001e200001009ae */
        /*1920*/                   ST.E [R2+0x56a0], R173 ;                           /* 0x000056a002007385 */
                                                                                      /* 0x0001e200001009ad */
        /*1930*/                   ST.E [R2+0x5620], R172 ;                           /* 0x0000562002007385 */
                                                                                      /* 0x0001e200001009ac */
        /*1940*/                   BRA.U 0x2350 ;                                     /* 0x00000a0100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1950*/                   ST.E [R2+0x5520], R170 ;                           /* 0x0000552002007385 */
                                                                                      /* 0x0001e200001009aa */
        /*1960*/                   ST.E [R2+0x54a0], R169 ;                           /* 0x000054a002007385 */
                                                                                      /* 0x0001e200001009a9 */
        /*1970*/                   ST.E [R2+0x5420], R168 ;                           /* 0x0000542002007385 */
                                                                                      /* 0x0001e200001009a8 */
        /*1980*/                   BRA.U 0x2350 ;                                     /* 0x000009c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1990*/                   ST.E [R2+0x5320], R166 ;                           /* 0x0000532002007385 */
                                                                                      /* 0x0001e200001009a6 */
        /*19a0*/                   ST.E [R2+0x52a0], R165 ;                           /* 0x000052a002007385 */
                                                                                      /* 0x0001e200001009a5 */
        /*19b0*/                   ST.E [R2+0x5220], R164 ;                           /* 0x0000522002007385 */
                                                                                      /* 0x0001e200001009a4 */
        /*19c0*/                   BRA.U 0x2350 ;                                     /* 0x0000098100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*19d0*/                   ST.E [R2+0x5120], R162 ;                           /* 0x0000512002007385 */
                                                                                      /* 0x0001e200001009a2 */
        /*19e0*/                   ST.E [R2+0x50a0], R161 ;                           /* 0x000050a002007385 */
                                                                                      /* 0x0001e200001009a1 */
        /*19f0*/                   ST.E [R2+0x5020], R160 ;                           /* 0x0000502002007385 */
                                                                                      /* 0x0001e200001009a0 */
        /*1a00*/                   BRA.U 0x2350 ;                                     /* 0x0000094100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1a10*/                   ST.E [R2+0x4f20], R158 ;                           /* 0x00004f2002007385 */
                                                                                      /* 0x0001e2000010099e */
        /*1a20*/                   ST.E [R2+0x4ea0], R157 ;                           /* 0x00004ea002007385 */
                                                                                      /* 0x0001e2000010099d */
        /*1a30*/                   ST.E [R2+0x4e20], R156 ;                           /* 0x00004e2002007385 */
                                                                                      /* 0x0001e2000010099c */
        /*1a40*/                   BRA.U 0x2350 ;                                     /* 0x0000090100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1a50*/                   ST.E [R2+0x4d20], R154 ;                           /* 0x00004d2002007385 */
                                                                                      /* 0x0001e2000010099a */
        /*1a60*/                   ST.E [R2+0x4ca0], R153 ;                           /* 0x00004ca002007385 */
                                                                                      /* 0x0001e20000100999 */
        /*1a70*/                   ST.E [R2+0x4c20], R152 ;                           /* 0x00004c2002007385 */
                                                                                      /* 0x0001e20000100998 */
        /*1a80*/                   BRA.U 0x2350 ;                                     /* 0x000008c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1a90*/                   ST.E [R2+0x4b20], R150 ;                           /* 0x00004b2002007385 */
                                                                                      /* 0x0001e20000100996 */
        /*1aa0*/                   ST.E [R2+0x4aa0], R149 ;                           /* 0x00004aa002007385 */
                                                                                      /* 0x0001e20000100995 */
        /*1ab0*/                   ST.E [R2+0x4a20], R148 ;                           /* 0x00004a2002007385 */
                                                                                      /* 0x0001e20000100994 */
        /*1ac0*/                   BRA.U 0x2350 ;                                     /* 0x0000088100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1ad0*/                   ST.E [R2+0x4920], R146 ;                           /* 0x0000492002007385 */
                                                                                      /* 0x0001e20000100992 */
        /*1ae0*/                   ST.E [R2+0x48a0], R145 ;                           /* 0x000048a002007385 */
                                                                                      /* 0x0001e20000100991 */
        /*1af0*/                   ST.E [R2+0x4820], R144 ;                           /* 0x0000482002007385 */
                                                                                      /* 0x0001e20000100990 */
        /*1b00*/                   BRA.U 0x2350 ;                                     /* 0x0000084100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1b10*/                   ST.E [R2+0x4720], R142 ;                           /* 0x0000472002007385 */
                                                                                      /* 0x0001e2000010098e */
        /*1b20*/                   ST.E [R2+0x46a0], R141 ;                           /* 0x000046a002007385 */
                                                                                      /* 0x0001e2000010098d */
        /*1b30*/                   ST.E [R2+0x4620], R140 ;                           /* 0x0000462002007385 */
                                                                                      /* 0x0001e2000010098c */
        /*1b40*/                   BRA.U 0x2350 ;                                     /* 0x0000080100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1b50*/                   ST.E [R2+0x4520], R138 ;                           /* 0x0000452002007385 */
                                                                                      /* 0x0001e2000010098a */
        /*1b60*/                   ST.E [R2+0x44a0], R137 ;                           /* 0x000044a002007385 */
                                                                                      /* 0x0001e20000100989 */
        /*1b70*/                   ST.E [R2+0x4420], R136 ;                           /* 0x0000442002007385 */
                                                                                      /* 0x0001e20000100988 */
        /*1b80*/                   BRA.U 0x2350 ;                                     /* 0x000007c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1b90*/                   ST.E [R2+0x4320], R134 ;                           /* 0x0000432002007385 */
                                                                                      /* 0x0001e20000100986 */
        /*1ba0*/                   ST.E [R2+0x42a0], R133 ;                           /* 0x000042a002007385 */
                                                                                      /* 0x0001e20000100985 */
        /*1bb0*/                   ST.E [R2+0x4220], R132 ;                           /* 0x0000422002007385 */
                                                                                      /* 0x0001e20000100984 */
        /*1bc0*/                   BRA.U 0x2350 ;                                     /* 0x0000078100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1bd0*/                   ST.E [R2+0x4120], R130 ;                           /* 0x0000412002007385 */
                                                                                      /* 0x0001e20000100982 */
        /*1be0*/                   ST.E [R2+0x40a0], R129 ;                           /* 0x000040a002007385 */
                                                                                      /* 0x0001e20000100981 */
        /*1bf0*/                   ST.E [R2+0x4020], R128 ;                           /* 0x0000402002007385 */
                                                                                      /* 0x0001e20000100980 */
        /*1c00*/                   BRA.U 0x2350 ;                                     /* 0x0000074100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1c10*/                   ST.E [R2+0x3f20], R126 ;                           /* 0x00003f2002007385 */
                                                                                      /* 0x0001e2000010097e */
        /*1c20*/                   ST.E [R2+0x3ea0], R125 ;                           /* 0x00003ea002007385 */
                                                                                      /* 0x0001e2000010097d */
        /*1c30*/                   ST.E [R2+0x3e20], R124 ;                           /* 0x00003e2002007385 */
                                                                                      /* 0x0001e2000010097c */
        /*1c40*/                   BRA.U 0x2350 ;                                     /* 0x0000070100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1c50*/                   ST.E [R2+0x3d20], R122 ;                           /* 0x00003d2002007385 */
                                                                                      /* 0x0001e2000010097a */
        /*1c60*/                   ST.E [R2+0x3ca0], R121 ;                           /* 0x00003ca002007385 */
                                                                                      /* 0x0001e20000100979 */
        /*1c70*/                   ST.E [R2+0x3c20], R120 ;                           /* 0x00003c2002007385 */
                                                                                      /* 0x0001e20000100978 */
        /*1c80*/                   BRA.U 0x2350 ;                                     /* 0x000006c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1c90*/                   ST.E [R2+0x3b20], R118 ;                           /* 0x00003b2002007385 */
                                                                                      /* 0x0001e20000100976 */
        /*1ca0*/                   ST.E [R2+0x3aa0], R117 ;                           /* 0x00003aa002007385 */
                                                                                      /* 0x0001e20000100975 */
        /*1cb0*/                   ST.E [R2+0x3a20], R116 ;                           /* 0x00003a2002007385 */
                                                                                      /* 0x0001e20000100974 */
        /*1cc0*/                   BRA.U 0x2350 ;                                     /* 0x0000068100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1cd0*/                   ST.E [R2+0x3920], R114 ;                           /* 0x0000392002007385 */
                                                                                      /* 0x0001e20000100972 */
        /*1ce0*/                   ST.E [R2+0x38a0], R113 ;                           /* 0x000038a002007385 */
                                                                                      /* 0x0001e20000100971 */
        /*1cf0*/                   ST.E [R2+0x3820], R112 ;                           /* 0x0000382002007385 */
                                                                                      /* 0x0001e20000100970 */
        /*1d00*/                   BRA.U 0x2350 ;                                     /* 0x0000064100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1d10*/                   ST.E [R2+0x3720], R110 ;                           /* 0x0000372002007385 */
                                                                                      /* 0x0001e2000010096e */
        /*1d20*/                   ST.E [R2+0x36a0], R109 ;                           /* 0x000036a002007385 */
                                                                                      /* 0x0001e2000010096d */
        /*1d30*/                   ST.E [R2+0x3620], R108 ;                           /* 0x0000362002007385 */
                                                                                      /* 0x0001e2000010096c */
        /*1d40*/                   BRA.U 0x2350 ;                                     /* 0x0000060100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1d50*/                   ST.E [R2+0x3520], R106 ;                           /* 0x0000352002007385 */
                                                                                      /* 0x0001e2000010096a */
        /*1d60*/                   ST.E [R2+0x34a0], R105 ;                           /* 0x000034a002007385 */
                                                                                      /* 0x0001e20000100969 */
        /*1d70*/                   ST.E [R2+0x3420], R104 ;                           /* 0x0000342002007385 */
                                                                                      /* 0x0001e20000100968 */
        /*1d80*/                   BRA.U 0x2350 ;                                     /* 0x000005c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1d90*/                   ST.E [R2+0x3320], R102 ;                           /* 0x0000332002007385 */
                                                                                      /* 0x0001e20000100966 */
        /*1da0*/                   ST.E [R2+0x32a0], R101 ;                           /* 0x000032a002007385 */
                                                                                      /* 0x0001e20000100965 */
        /*1db0*/                   ST.E [R2+0x3220], R100 ;                           /* 0x0000322002007385 */
                                                                                      /* 0x0001e20000100964 */
        /*1dc0*/                   BRA.U 0x2350 ;                                     /* 0x0000058100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1dd0*/                   ST.E [R2+0x3120], R98 ;                            /* 0x0000312002007385 */
                                                                                      /* 0x0001e20000100962 */
        /*1de0*/                   ST.E [R2+0x30a0], R97 ;                            /* 0x000030a002007385 */
                                                                                      /* 0x0001e20000100961 */
        /*1df0*/                   ST.E [R2+0x3020], R96 ;                            /* 0x0000302002007385 */
                                                                                      /* 0x0001e20000100960 */
        /*1e00*/                   BRA.U 0x2350 ;                                     /* 0x0000054100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1e10*/                   ST.E [R2+0x2f20], R94 ;                            /* 0x00002f2002007385 */
                                                                                      /* 0x0001e2000010095e */
        /*1e20*/                   ST.E [R2+0x2ea0], R93 ;                            /* 0x00002ea002007385 */
                                                                                      /* 0x0001e2000010095d */
        /*1e30*/                   ST.E [R2+0x2e20], R92 ;                            /* 0x00002e2002007385 */
                                                                                      /* 0x0001e2000010095c */
        /*1e40*/                   BRA.U 0x2350 ;                                     /* 0x0000050100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1e50*/                   ST.E [R2+0x2d20], R90 ;                            /* 0x00002d2002007385 */
                                                                                      /* 0x0001e2000010095a */
        /*1e60*/                   ST.E [R2+0x2ca0], R89 ;                            /* 0x00002ca002007385 */
                                                                                      /* 0x0001e20000100959 */
        /*1e70*/                   ST.E [R2+0x2c20], R88 ;                            /* 0x00002c2002007385 */
                                                                                      /* 0x0001e20000100958 */
        /*1e80*/                   BRA.U 0x2350 ;                                     /* 0x000004c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1e90*/                   ST.E [R2+0x2b20], R86 ;                            /* 0x00002b2002007385 */
                                                                                      /* 0x0001e20000100956 */
        /*1ea0*/                   ST.E [R2+0x2aa0], R85 ;                            /* 0x00002aa002007385 */
                                                                                      /* 0x0001e20000100955 */
        /*1eb0*/                   ST.E [R2+0x2a20], R84 ;                            /* 0x00002a2002007385 */
                                                                                      /* 0x0001e20000100954 */
        /*1ec0*/                   BRA.U 0x2350 ;                                     /* 0x0000048100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1ed0*/                   ST.E [R2+0x2920], R82 ;                            /* 0x0000292002007385 */
                                                                                      /* 0x0001e20000100952 */
        /*1ee0*/                   ST.E [R2+0x28a0], R81 ;                            /* 0x000028a002007385 */
                                                                                      /* 0x0001e20000100951 */
        /*1ef0*/                   ST.E [R2+0x2820], R80 ;                            /* 0x0000282002007385 */
                                                                                      /* 0x0001e20000100950 */
        /*1f00*/                   BRA.U 0x2350 ;                                     /* 0x0000044100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1f10*/                   ST.E [R2+0x2720], R78 ;                            /* 0x0000272002007385 */
                                                                                      /* 0x0001e2000010094e */
        /*1f20*/                   ST.E [R2+0x26a0], R77 ;                            /* 0x000026a002007385 */
                                                                                      /* 0x0001e2000010094d */
        /*1f30*/                   ST.E [R2+0x2620], R76 ;                            /* 0x0000262002007385 */
                                                                                      /* 0x0001e2000010094c */
        /*1f40*/                   BRA.U 0x2350 ;                                     /* 0x0000040100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1f50*/                   ST.E [R2+0x2520], R74 ;                            /* 0x0000252002007385 */
                                                                                      /* 0x0001e2000010094a */
        /*1f60*/                   ST.E [R2+0x24a0], R73 ;                            /* 0x000024a002007385 */
                                                                                      /* 0x0001e20000100949 */
        /*1f70*/                   ST.E [R2+0x2420], R72 ;                            /* 0x0000242002007385 */
                                                                                      /* 0x0001e20000100948 */
        /*1f80*/                   BRA.U 0x2350 ;                                     /* 0x000003c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1f90*/                   ST.E [R2+0x2320], R70 ;                            /* 0x0000232002007385 */
                                                                                      /* 0x0001e20000100946 */
        /*1fa0*/                   ST.E [R2+0x22a0], R69 ;                            /* 0x000022a002007385 */
                                                                                      /* 0x0001e20000100945 */
        /*1fb0*/                   ST.E [R2+0x2220], R68 ;                            /* 0x0000222002007385 */
                                                                                      /* 0x0001e20000100944 */
        /*1fc0*/                   BRA.U 0x2350 ;                                     /* 0x0000038100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*1fd0*/                   ST.E [R2+0x2120], R66 ;                            /* 0x0000212002007385 */
                                                                                      /* 0x0001e20000100942 */
        /*1fe0*/                   ST.E [R2+0x20a0], R65 ;                            /* 0x000020a002007385 */
                                                                                      /* 0x0001e20000100941 */
        /*1ff0*/                   ST.E [R2+0x2020], R64 ;                            /* 0x0000202002007385 */
                                                                                      /* 0x0001e20000100940 */
        /*2000*/                   BRA.U 0x2350 ;                                     /* 0x0000034100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2010*/                   ST.E [R2+0x1f20], R62 ;                            /* 0x00001f2002007385 */
                                                                                      /* 0x0001e2000010093e */
        /*2020*/                   ST.E [R2+0x1ea0], R61 ;                            /* 0x00001ea002007385 */
                                                                                      /* 0x0001e2000010093d */
        /*2030*/                   ST.E [R2+0x1e20], R60 ;                            /* 0x00001e2002007385 */
                                                                                      /* 0x0001e2000010093c */
        /*2040*/                   BRA.U 0x2350 ;                                     /* 0x0000030100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2050*/                   ST.E [R2+0x1d20], R58 ;                            /* 0x00001d2002007385 */
                                                                                      /* 0x0001e2000010093a */
        /*2060*/                   ST.E [R2+0x1ca0], R57 ;                            /* 0x00001ca002007385 */
                                                                                      /* 0x0001e20000100939 */
        /*2070*/                   ST.E [R2+0x1c20], R56 ;                            /* 0x00001c2002007385 */
                                                                                      /* 0x0001e20000100938 */
        /*2080*/                   BRA.U 0x2350 ;                                     /* 0x000002c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2090*/                   ST.E [R2+0x1b20], R54 ;                            /* 0x00001b2002007385 */
                                                                                      /* 0x0001e20000100936 */
        /*20a0*/                   ST.E [R2+0x1aa0], R53 ;                            /* 0x00001aa002007385 */
                                                                                      /* 0x0001e20000100935 */
        /*20b0*/                   ST.E [R2+0x1a20], R52 ;                            /* 0x00001a2002007385 */
                                                                                      /* 0x0001e20000100934 */
        /*20c0*/                   BRA.U 0x2350 ;                                     /* 0x0000028100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*20d0*/                   ST.E [R2+0x1920], R50 ;                            /* 0x0000192002007385 */
                                                                                      /* 0x0001e20000100932 */
        /*20e0*/                   ST.E [R2+0x18a0], R49 ;                            /* 0x000018a002007385 */
                                                                                      /* 0x0001e20000100931 */
        /*20f0*/                   ST.E [R2+0x1820], R48 ;                            /* 0x0000182002007385 */
                                                                                      /* 0x0001e20000100930 */
        /*2100*/                   BRA.U 0x2350 ;                                     /* 0x0000024100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2110*/                   ST.E [R2+0x1720], R46 ;                            /* 0x0000172002007385 */
                                                                                      /* 0x0001e2000010092e */
        /*2120*/                   ST.E [R2+0x16a0], R45 ;                            /* 0x000016a002007385 */
                                                                                      /* 0x0001e2000010092d */
        /*2130*/                   ST.E [R2+0x1620], R44 ;                            /* 0x0000162002007385 */
                                                                                      /* 0x0001e2000010092c */
        /*2140*/                   BRA.U 0x2350 ;                                     /* 0x0000020100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2150*/                   ST.E [R2+0x1520], R42 ;                            /* 0x0000152002007385 */
                                                                                      /* 0x0001e2000010092a */
        /*2160*/                   ST.E [R2+0x14a0], R41 ;                            /* 0x000014a002007385 */
                                                                                      /* 0x0001e20000100929 */
        /*2170*/                   ST.E [R2+0x1420], R40 ;                            /* 0x0000142002007385 */
                                                                                      /* 0x0001e20000100928 */
        /*2180*/                   BRA.U 0x2350 ;                                     /* 0x000001c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2190*/                   ST.E [R2+0x1320], R38 ;                            /* 0x0000132002007385 */
                                                                                      /* 0x0001e20000100926 */
        /*21a0*/                   ST.E [R2+0x12a0], R37 ;                            /* 0x000012a002007385 */
                                                                                      /* 0x0001e20000100925 */
        /*21b0*/                   ST.E [R2+0x1220], R36 ;                            /* 0x0000122002007385 */
                                                                                      /* 0x0001e20000100924 */
        /*21c0*/                   BRA.U 0x2350 ;                                     /* 0x0000018100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*21d0*/                   ST.E [R2+0x1120], R34 ;                            /* 0x0000112002007385 */
                                                                                      /* 0x0001e20000100922 */
        /*21e0*/                   ST.E [R2+0x10a0], R33 ;                            /* 0x000010a002007385 */
                                                                                      /* 0x0001e20000100921 */
        /*21f0*/                   ST.E [R2+0x1020], R32 ;                            /* 0x0000102002007385 */
                                                                                      /* 0x0001e20000100920 */
        /*2200*/                   BRA.U 0x2350 ;                                     /* 0x0000014100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2210*/                   ST.E [R2+0xf20], R30 ;                             /* 0x00000f2002007385 */
                                                                                      /* 0x0001e2000010091e */
        /*2220*/                   ST.E [R2+0xea0], R29 ;                             /* 0x00000ea002007385 */
                                                                                      /* 0x0001e2000010091d */
        /*2230*/                   ST.E [R2+0xe20], R28 ;                             /* 0x00000e2002007385 */
                                                                                      /* 0x0001e2000010091c */
        /*2240*/                   BRA.U 0x2350 ;                                     /* 0x0000010100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2250*/                   ST.E [R2+0xd20], R26 ;                             /* 0x00000d2002007385 */
                                                                                      /* 0x0001e2000010091a */
        /*2260*/                   ST.E [R2+0xca0], R25 ;                             /* 0x00000ca002007385 */
                                                                                      /* 0x0001e20000100919 */
        /*2270*/                   ST.E [R2+0xc20], R24 ;                             /* 0x00000c2002007385 */
                                                                                      /* 0x0001e20000100918 */
        /*2280*/                   BRA.U 0x2350 ;                                     /* 0x000000c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2290*/                   ST.E [R2+0xb20], R22 ;                             /* 0x00000b2002007385 */
                                                                                      /* 0x0001e20000100916 */
        /*22a0*/                   ST.E [R2+0xaa0], R21 ;                             /* 0x00000aa002007385 */
                                                                                      /* 0x0001e20000100915 */
        /*22b0*/                   ST.E [R2+0xa20], R20 ;                             /* 0x00000a2002007385 */
                                                                                      /* 0x0001e20000100914 */
        /*22c0*/                   BRA.U 0x2350 ;                                     /* 0x0000008100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*22d0*/                   ST.E [R2+0x920], R18 ;                             /* 0x0000092002007385 */
                                                                                      /* 0x0001e20000100912 */
        /*22e0*/                   ST.E [R2+0x8a0], R17 ;                             /* 0x000008a002007385 */
                                                                                      /* 0x0001e20000100911 */
        /*22f0*/                   ST.E [R2+0x820], R16 ;                             /* 0x0000082002007385 */
                                                                                      /* 0x0001e20000100910 */
        /*2300*/                   BRA.U 0x2350 ;                                     /* 0x0000004100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2310*/                   ST.E [R2+0x720], R14 ;                             /* 0x0000072002007385 */
                                                                                      /* 0x0001e2000010090e */
        /*2320*/                   ST.E [R2+0x6a0], R9 ;                              /* 0x000006a002007385 */
                                                                                      /* 0x0201e20000100909 */
        /*2330*/                   ST.E [R2+0x620], R8 ;                              /* 0x0000062002007385 */
                                                                                      /* 0x0201e20000100908 */
        /*2340*/                   BRA.U 0x2350 ;                                     /* 0x0000000100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2350*/                   MOV R2, 0xf6000000 ;                               /* 0xf600000000027802 */
                                                                                      /* 0x001fe40000000f00 */
        /*2360*/                   MOV R3, 0x7f4a ;                                   /* 0x00007f4a00037802 */
                                                                                      /* 0x001fe20000000f00 */
        /*2370*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x001ee20000004300 */
        /*2380*/                   MOV R5, 0x74a20 ;                                  /* 0x00074a2000057802 */
                                                                                      /* 0x000fec0000000f00 */
        /*2390*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x008fea00078e0002 */
        /*23a0*/                   IADD3 R2, P3, R2, 0x69020, RZ ;                    /* 0x0006902002027810 */
                                                                                      /* 0x000fea0007f7e0ff */
        /*23b0*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x000fe20001ffe4ff */
        /*23c0*/                   S2R R4, SR_VIRTID ;                                /* 0x0000000000047919 */
                                                                                      /* 0x000ee40000000300 */
        /*23d0*/                   SHF.R.U32.HI R4, RZ, 0x8, R4 ;                     /* 0x00000008ff047819 */
                                                                                      /* 0x008fec0000011604 */
        /*23e0*/                   SGXT.U32 R4, R4, 0x7 ;                             /* 0x000000070404781a */
                                                                                      /* 0x000fe40000000000 */
        /*23f0*/                   MOV R5, 0x2c0 ;                                    /* 0x000002c000057802 */
                                                                                      /* 0x000fec0000000f00 */
        /*2400*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x000fea00078e0002 */
        /*2410*/                   ST.E [R2+0x294], R0 ;                              /* 0x0000029402007385 */
                                                                                      /* 0x0083e40000100900 */
        /*2420*/                   MOV R2, 0xf6000000 ;                               /* 0xf600000000027802 */
                                                                                      /* 0x002fe40000000f00 */
        /*2430*/                   MOV R3, 0x7f4a ;                                   /* 0x00007f4a00037802 */
                                                                                      /* 0x002fe20000000f00 */
        /*2440*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x000e220000004300 */
        /*2450*/                   MOV R5, 0x74a20 ;                                  /* 0x00074a2000057802 */
                                                                                      /* 0x000fec0000000f00 */
        /*2460*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x003fea00078e0002 */
        /*2470*/                   IADD3 R2, P3, R2, 0x74020, RZ ;                    /* 0x0007402002027810 */
                                                                                      /* 0x002fea0007f7e0ff */
        /*2480*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x002fe20001ffe4ff */
        /*2490*/                   S2R R4, SR_VIRTID ;                                /* 0x0000000000047919 */
                                                                                      /* 0x000e240000000300 */
        /*24a0*/                   LOP3.LUT R5, R4, 0xf0000, RZ, 0xc0, !PT ;          /* 0x000f000004057812 */
                                                                                      /* 0x001fec00078ec0ff */
        /*24b0*/                   SHF.R.S32.HI R5, RZ, 0x10, R5 ;                    /* 0x00000010ff057819 */
                                                                                      /* 0x000fe40000011405 */
        /*24c0*/                   LOP3.LUT R4, R4, 0x60000000, RZ, 0xc0, !PT ;       /* 0x6000000004047812 */
                                                                                      /* 0x000fec00078ec0ff */
        /*24d0*/                   SHF.R.S32.HI R4, RZ, 0x19, R4 ;                    /* 0x00000019ff047819 */
                                                                                      /* 0x000fec0000011404 */
        /*24e0*/                   LOP3.LUT R4, R4, R5, RZ, 0xfc, !PT ;               /* 0x0000000504047212 */
                                                                                      /* 0x000fe400078efcff */
        /*24f0*/                   MOV R5, 0x50 ;                                     /* 0x0000005000057802 */
                                                                                      /* 0x000fec0000000f00 */
        /*2500*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x002fe400078e0002 */
        /*2510*/               @P2 S2R R4, SR_CTAID.X ;                               /* 0x0000000000042919 */
                                                                                      /* 0x000ee20000002500 */
        /*2520*/               @P2 S2R R5, SR_CTAID.Y ;                               /* 0x0000000000052919 */
                                                                                      /* 0x000e220000002600 */
        /*2530*/               @P2 S2R R6, SR_CTAID.Z ;                               /* 0x0000000000062919 */
                                                                                      /* 0x004ea40000002700 */
        /*2540*/               @P2 SHF.L.U32 R6, R6, 0x10, RZ ;                       /* 0x0000001006062819 */
                                                                                      /* 0x004fe400000006ff */
        /*2550*/               @P2 LOP3.LUT R5, R5, 0xffff, RZ, 0xc0, !PT ;           /* 0x0000ffff05052812 */
                                                                                      /* 0x001fec00078ec0ff */
        /*2560*/               @P2 LOP3.LUT R5, R5, R6, RZ, 0xfc, !PT ;               /* 0x0000000605052212 */
                                                                                      /* 0x000fea00078efcff */
        /*2570*/               @P2 ST.E.64 [R2], R4 ;                                 /* 0x0000000002002385 */
                                                                                      /* 0x0083e40000100b04 */
        /*2580*/               @P2 B2R R4, 0x0 ;                                      /* 0x000000000004231c */
                                                                                      /* 0x002f220000000000 */
        /*2590*/               @P2 B2R R5, 0x1 ;                                      /* 0x004000000005231c */
                                                                                      /* 0x002e220000000000 */
        /*25a0*/               @P2 B2R R6, 0x2 ;                                      /* 0x008000000006231c */
                                                                                      /* 0x000ee20000000000 */
        /*25b0*/               @P2 B2R R7, 0x3 ;                                      /* 0x00c000000007231c */
                                                                                      /* 0x000ea40000000000 */
        /*25c0*/               @P2 ST.E.128 [R2+0x10], R4 ;                           /* 0x0000001002002385 */
                                                                                      /* 0x01d3e40000100d04 */
        /*25d0*/               @P2 B2R R4, 0x4 ;                                      /* 0x010000000004231c */
                                                                                      /* 0x002f220000000000 */
        /*25e0*/               @P2 B2R R5, 0x5 ;                                      /* 0x014000000005231c */
                                                                                      /* 0x002e220000000000 */
        /*25f0*/               @P2 B2R R6, 0x6 ;                                      /* 0x018000000006231c */
                                                                                      /* 0x002ee20000000000 */
        /*2600*/               @P2 B2R R7, 0x7 ;                                      /* 0x01c000000007231c */
                                                                                      /* 0x002ea40000000000 */
        /*2610*/               @P2 ST.E.128 [R2+0x20], R4 ;                           /* 0x0000002002002385 */
                                                                                      /* 0x01d3e40000100d04 */
        /*2620*/               @P2 B2R R4, 0x8 ;                                      /* 0x020000000004231c */
                                                                                      /* 0x002f220000000000 */
        /*2630*/               @P2 B2R R5, 0x9 ;                                      /* 0x024000000005231c */
                                                                                      /* 0x002e220000000000 */
        /*2640*/               @P2 B2R R6, 0xa ;                                      /* 0x028000000006231c */
                                                                                      /* 0x002ee20000000000 */
        /*2650*/               @P2 B2R R7, 0xb ;                                      /* 0x02c000000007231c */
                                                                                      /* 0x002ea40000000000 */
        /*2660*/               @P2 ST.E.128 [R2+0x30], R4 ;                           /* 0x0000003002002385 */
                                                                                      /* 0x01d3e40000100d04 */
        /*2670*/               @P2 B2R R4, 0xc ;                                      /* 0x030000000004231c */
                                                                                      /* 0x002ea20000000000 */
        /*2680*/               @P2 B2R R5, 0xd ;                                      /* 0x034000000005231c */
                                                                                      /* 0x002ea20000000000 */
        /*2690*/               @P2 B2R R6, 0xe ;                                      /* 0x038000000006231c */
                                                                                      /* 0x002ea20000000000 */
        /*26a0*/               @P2 B2R R7, 0xf ;                                      /* 0x03c000000007231c */
                                                                                      /* 0x002ea40000000000 */
        /*26b0*/               @P2 ST.E.128 [R2+0x40], R4 ;                           /* 0x0000004002002385 */
                                                                                      /* 0x0043e40000100d04 */
        /*26c0*/                   S2R R11, SR_SMEMSZ ;                               /* 0x00000000000b7919 */
                                                                                      /* 0x003ee40000003200 */
        /*26d0*/                   ISETP.EQ.AND P6, PT, R11, RZ, PT ;                 /* 0x000000ff0b00720c */
                                                                                      /* 0x008fda0003fc2270 */
        /*26e0*/               @P6 BRA.U 0x2c70 ;                                     /* 0x0000058100006947 */
                                                                                      /* 0x000fea0003800000 */
        /*26f0*/                   MOV R2, 0xf6000000 ;                               /* 0xf600000000027802 */
                                                                                      /* 0x002fe40000000f00 */
        /*2700*/                   MOV R3, 0x7f4a ;                                   /* 0x00007f4a00037802 */
                                                                                      /* 0x002fe20000000f00 */
        /*2710*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x002ea20000004300 */
        /*2720*/                   MOV R5, 0x74a20 ;                                  /* 0x00074a2000057802 */
                                                                                      /* 0x002fec0000000f00 */
        /*2730*/                   IMAD.WIDE.U32 R2, R4, R5, R2 ;                     /* 0x0000000504027225 */
                                                                                      /* 0x006fe200078e0002 */
        /*2740*/                   MOV R0, 0x0 ;                                      /* 0x0000000000007802 */
                                                                                      /* 0x002fec0000000f00 */
        /*2750*/                   ISETP.NE.AND P6, PT, R0, RZ, PT ;                  /* 0x000000ff0000720c */
                                                                                      /* 0x000fda0003fc5270 */
        /*2760*/               @P6 BRA.U 0x2910 ;                                     /* 0x000001a100006947 */
                                                                                      /* 0x000fea0003800000 */
        /*2770*/               @P2 ATOM.E.ADD PT, R0, [R2+0x4], R11 ;                 /* 0x0000040b0200238a */
                                                                                      /* 0x00222400001e0100 */
        /*2780*/                   MOV R8, 0xf6000000 ;                               /* 0xf600000000087802 */
                                                                                      /* 0x021fe40000000f00 */
        /*2790*/                   MOV R9, 0x7f4a ;                                   /* 0x00007f4a00097802 */
                                                                                      /* 0x021fe20000000f00 */
        /*27a0*/                   S2R R4, SR_VIRTUALSMID ;                           /* 0x0000000000047919 */
                                                                                      /* 0x002ea20000004300 */
        /*27b0*/                   MOV R5, 0x74a20 ;                                  /* 0x00074a2000057802 */
                                                                                      /* 0x002fec0000000f00 */
        /*27c0*/                   IMAD.WIDE.U32 R8, R4, R5, R8 ;                     /* 0x0000000504087225 */
                                                                                      /* 0x004fea00078e0008 */
        /*27d0*/                   IADD3 R8, P3, R8, 0x74020, RZ ;                    /* 0x0007402008087810 */
                                                                                      /* 0x000fea0007f7e0ff */
        /*27e0*/                   IADD3.X R9, R9, RZ, RZ, P3, !PT ;                  /* 0x000000ff09097210 */
                                                                                      /* 0x000fe20001ffe4ff */
        /*27f0*/                   S2R R4, SR_VIRTID ;                                /* 0x0000000000047919 */
                                                                                      /* 0x002ea40000000300 */
        /*2800*/                   LOP3.LUT R5, R4, 0xf0000, RZ, 0xc0, !PT ;          /* 0x000f000004057812 */
                                                                                      /* 0x006fec00078ec0ff */
        /*2810*/                   SHF.R.S32.HI R5, RZ, 0x10, R5 ;                    /* 0x00000010ff057819 */
                                                                                      /* 0x002fe40000011405 */
        /*2820*/                   LOP3.LUT R4, R4, 0x60000000, RZ, 0xc0, !PT ;       /* 0x6000000004047812 */
                                                                                      /* 0x002fec00078ec0ff */
        /*2830*/                   SHF.R.S32.HI R4, RZ, 0x19, R4 ;                    /* 0x00000019ff047819 */
                                                                                      /* 0x002fec0000011404 */
        /*2840*/                   LOP3.LUT R4, R4, R5, RZ, 0xfc, !PT ;               /* 0x0000000504047212 */
                                                                                      /* 0x002fe400078efcff */
        /*2850*/                   MOV R5, 0x50 ;                                     /* 0x0000005000057802 */
                                                                                      /* 0x002fec0000000f00 */
        /*2860*/                   IMAD.WIDE.U32 R8, R4, R5, R8 ;                     /* 0x0000000504087225 */
                                                                                      /* 0x000fea00078e0008 */
        /*2870*/               @P2 ST.E [R8+0x8], R0 ;                                /* 0x0000000808002385 */
                                                                                      /* 0x0013e40000100900 */
        /*2880*/               @P2 ATOMS.EXCH R4, [RZ], R0 ;                          /* 0x00000000ff04238c */
                                                                                      /* 0x0022a20004000000 */
        /*2890*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*28a0*/              @!P2 LDS R0, [RZ] ;                                     /* 0x00000000ff00a984 */
                                                                                      /* 0x002e220000000800 */
        /*28b0*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*28c0*/               @P2 STS [RZ], R4 ;                                     /* 0x00000004ff002388 */
                                                                                      /* 0x0043e20000000800 */
        /*28d0*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*28e0*/                   IADD3 R2, P3, R2, R0, RZ ;                         /* 0x0000000002027210 */
                                                                                      /* 0x003fea0007f7e0ff */
        /*28f0*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x002fe20001ffe4ff */
        /*2900*/                   BRA.U 0x29d0 ;                                     /* 0x000000c100007947 */
                                                                                      /* 0x000fea0003800000 */
        /*2910*/                   S2R R0, SR_VIRTID ;                                /* 0x0000000000007919 */
                                                                                      /* 0x002e240000000300 */
        /*2920*/                   LOP3.LUT R4, R0, 0xf0000, RZ, 0xc0, !PT ;          /* 0x000f000000047812 */
                                                                                      /* 0x007fec00078ec0ff */
        /*2930*/                   SHF.R.S32.HI R4, RZ, 0x10, R4 ;                    /* 0x00000010ff047819 */
                                                                                      /* 0x000fe40000011404 */
        /*2940*/                   LOP3.LUT R0, R0, 0x60000000, RZ, 0xc0, !PT ;       /* 0x6000000000007812 */
                                                                                      /* 0x000fec00078ec0ff */
        /*2950*/                   SHF.R.S32.HI R0, RZ, 0x19, R0 ;                    /* 0x00000019ff007819 */
                                                                                      /* 0x000fec0000011400 */
        /*2960*/                   LOP3.LUT R0, R0, R4, RZ, 0xfc, !PT ;               /* 0x0000000400007212 */
                                                                                      /* 0x000fec00078efcff */
        /*2970*/                   IADD3 R0, R0, 0x1, RZ ;                            /* 0x0000000100007810 */
                                                                                      /* 0x000fea0007ffe0ff */
        /*2980*/               @P2 ATOM.E.CAS PT, R4, [R2+0x4], RZ, R0 ;              /* 0x000004ff0204238b */
                                                                                      /* 0x0002a200001e0100 */
        /*2990*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*29a0*/                   LD.E R4, [R2+0x4] ;                                /* 0x0000000402047980 */
                                                                                      /* 0x0042a40000100900 */
        /*29b0*/                   ISETP.EQ.AND P6, PT, R0, R4, PT ;                  /* 0x000000040000720c */
                                                                                      /* 0x004fda0003fc2270 */
        /*29c0*/              @!P6 BRA.U 0x2c70 ;                                     /* 0x000002a10000e947 */
                                                                                      /* 0x000fea0003800000 */
        /*29d0*/                   S2R R0, SR_NLATC ;                                 /* 0x0000000000007919 */
                                                                                      /* 0x003e240000002a00 */
        /*29e0*/                   ISETP.GT.AND P6, PT, R0, 0x20, PT ;                /* 0x000000200000780c */
                                                                                      /* 0x001fda0003fc4270 */
        /*29f0*/               @P6 BRA.U 0x2ad0 ;                                     /* 0x000000d100006947 */
                                                                                      /* 0x000fea0003800000 */
        /*2a00*/                   S2R R10, SR_LANEID ;                               /* 0x00000000000a7919 */
                                                                                      /* 0x003e240000000000 */
        /*2a10*/                   IMAD.SHL R0, R10, 0x10, RZ ;                       /* 0x000000100a007824 */
                                                                                      /* 0x001fea00078e02ff */
        /*2a20*/                   IADD3 R2, P3, R2, R0, RZ ;                         /* 0x0000000002027210 */
                                                                                      /* 0x002fea0007f7e0ff */
        /*2a30*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x002fe40001ffe4ff */
        /*2a40*/                   ISETP.LT.AND P6, PT, R0, R11, PT ;                 /* 0x0000000b0000720c */
                                                                                      /* 0x008fda0003fc1270 */
        /*2a50*/               @P6 LDS.128 R4, [R0] ;                                 /* 0x0000000000046984 */
                                                                                      /* 0x0060a40000000c00 */
        /*2a60*/               @P6 ST.E.128 [R2+0x40020], R4 ;                        /* 0x0004002002006385 */
                                                                                      /* 0x0043e20000100d04 */
        /*2a70*/                   IADD3 R0, R0, 0x200, RZ ;                          /* 0x0000020000007810 */
                                                                                      /* 0x001fe40007ffe0ff */
        /*2a80*/                   IADD3 R2, P3, R2, 0x200, RZ ;                      /* 0x0000020002027810 */
                                                                                      /* 0x002fea0007f7e0ff */
        /*2a90*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x002fe40001ffe4ff */
        /*2aa0*/                   ISETP.LT.AND P6, PT, R0, R11, PT ;                 /* 0x0000000b0000720c */
                                                                                      /* 0x008fda0003fc1270 */
        /*2ab0*/              @!P6 BRA.U 0x2c70 ;                                     /* 0x000001b10000e947 */
                                                                                      /* 0x000fea0003800000 */
        /*2ac0*/                   BRA 0x2a50 ;                                       /* 0xffffff8000007947 */
                                                                                      /* 0x000fea000383ffff */
        /*2ad0*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*2ae0*/                   R2B 0x0, RZ ;                                      /* 0x000000ff0000731e */
                                                                                      /* 0x000fe20000000000 */
        /*2af0*/                   R2B.WARP 0x0, RZ ;                                 /* 0x000000ff0000731e */
                                                                                      /* 0x000fe20000008000 */
        /*2b00*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*2b10*/                   MOV R4, 0x600 ;                                    /* 0x0000060000047802 */
                                                                                      /* 0x006fea0000000f00 */
        /*2b20*/                   BAR.SCAN 0x0, R4, PT ;                             /* 0x000000040000791d */
                                                                                      /* 0x0003ec0003806000 */
        /*2b30*/                   B2R.RESULT R6 ;                                    /* 0x000000000006731c */
                                                                                      /* 0x006ea200000e4000 */
        /*2b40*/                   S2R R11, SR_LANEID ;                               /* 0x00000000000b7919 */
                                                                                      /* 0x00aea40000000000 */
        /*2b50*/                   IADD3 R11, R11, R6, RZ ;                           /* 0x000000060b0b7210 */
                                                                                      /* 0x004fe80007ffe0ff */
        /*2b60*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*2b70*/                   S2R R4, SR_NLATC ;                                 /* 0x0000000000047919 */
                                                                                      /* 0x002ea20000002a00 */
        /*2b80*/                   IMAD.SHL R0, R11, 0x10, RZ ;                       /* 0x000000100b007824 */
                                                                                      /* 0x001fea00078e02ff */
        /*2b90*/                   IADD3 R2, P3, R2, R0, RZ ;                         /* 0x0000000002027210 */
                                                                                      /* 0x002fea0007f7e0ff */
        /*2ba0*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x002fe20001ffe4ff */
        /*2bb0*/                   IMAD.SHL R11, R4, 0x10, RZ ;                       /* 0x00000010040b7824 */
                                                                                      /* 0x004fe400078e02ff */
        /*2bc0*/                   S2R R4, SR_SMEMSZ ;                                /* 0x0000000000047919 */
                                                                                      /* 0x002ea40000003200 */
        /*2bd0*/                   ISETP.LT.AND P6, PT, R0, R4, PT ;                  /* 0x000000040000720c */
                                                                                      /* 0x004fda0003fc1270 */
        /*2be0*/               @P6 LDS.128 R4, [R0] ;                                 /* 0x0000000000046984 */
                                                                                      /* 0x0060a40000000c00 */
        /*2bf0*/               @P6 ST.E.128 [R2+0x40020], R4 ;                        /* 0x0004002002006385 */
                                                                                      /* 0x0043e20000100d04 */
        /*2c00*/                   IADD3 R0, R0, R11, RZ ;                            /* 0x0000000b00007210 */
                                                                                      /* 0x001fe40007ffe0ff */
        /*2c10*/                   IADD3 R2, P3, R2, R11, RZ ;                        /* 0x0000000b02027210 */
                                                                                      /* 0x002fea0007f7e0ff */
        /*2c20*/                   IADD3.X R3, R3, RZ, RZ, P3, !PT ;                  /* 0x000000ff03037210 */
                                                                                      /* 0x002fe20001ffe4ff */
        /*2c30*/                   S2R R4, SR_SMEMSZ ;                                /* 0x0000000000047919 */
                                                                                      /* 0x002ea40000003200 */
        /*2c40*/                   ISETP.LT.AND P6, PT, R0, R4, PT ;                  /* 0x000000040000720c */
                                                                                      /* 0x004fda0003fc1270 */
        /*2c50*/              @!P6 BRA.U 0x2c70 ;                                     /* 0x000000110000e947 */
                                                                                      /* 0x000fea0003800000 */
        /*2c60*/                   BRA 0x2be0 ;                                       /* 0xffffff7000007947 */
                                                                                      /* 0x000fea000383ffff */
        /*2c70*/                   ISETP.EQ.AND P4, PT, RZ, RZ, PT ;                  /* 0x000000ffff00720c */
                                                                                      /* 0x000fe20003f82270 */
        /*2c80*/                   S2R R6, SR_SW_SCRATCH ;                            /* 0x0000000000067919 */
                                                                                      /* 0x006e240000001800 */
        /*2c90*/                   LOP3.LUT R6, R6, 0x1, RZ, 0xc0, !PT ;              /* 0x0000000106067812 */
                                                                                      /* 0x001fec00078ec0ff */
        /*2ca0*/                   ISETP.EQ.XOR P4, PT, R6, RZ, P4 ;                  /* 0x000000ff0600720c */
                                                                                      /* 0x000fee0002782a70 */
        /*2cb0*/                   MOV R4, 0x13800000 ;                               /* 0x1380000000047802 */
                                                                                      /* 0x006fe80000000f00 */
        /*2cc0*/                   MOV R5, 0x7f4b ;                                   /* 0x00007f4b00057802 */
                                                                                      /* 0x006fe40000000f00 */
        /*2cd0*/               @P4 IADD3 R4, P3, R4, 0x4, RZ ;                        /* 0x0000000404044810 */
                                                                                      /* 0x000fea0007f7e0ff */
        /*2ce0*/               @P4 IADD3.X R5, R5, RZ, RZ, P3, !PT ;                  /* 0x000000ff05054210 */
                                                                                      /* 0x000fe40001ffe4ff */
        /*2cf0*/                   MOV R6, 0x1 ;                                      /* 0x0000000100067802 */
                                                                                      /* 0x000fea0000000f00 */
        /*2d00*/               @P2 RED.E.ADD [R4], R6 ;                               /* 0x000000060400298e */
                                                                                      /* 0x000fe20000100100 */
        /*2d10*/                   BAR.SYNCALL ;                                      /* 0x0000000000007b1d */
                                                                                      /* 0x000fec0000008000 */
        /*2d20*/                   RET.ABS R12 0x20;                                  /* 0x000000200c007950 */
                                                                                      /* 0x000fc00003a00000 */
        /*2d30*/                   BRA 0x2d30;                                        /* 0xfffffff000007947 */
                                                                                      /* 0x000fc0000383ffff */
        /*2d40*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2d50*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2d60*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2d70*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2d80*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2d90*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2da0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2db0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2dc0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2dd0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2de0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*2df0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */


	code for sm_86
                Entry Point : 0x7fac32f6fb00
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   MOV R4, 0x33400000 ;                               /* 0x3340000000047802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0010*/                   MOV R5, 0x7fac ;                                   /* 0x00007fac00057802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0020*/                   MOV R0, 0x1 ;                                      /* 0x0000000100007802 */
                                                                                      /* 0x000fea0000000f00 */
        /*0030*/                   ST.E.STRONG.SM.PRIVATE [R4], R0 ;                  /* 0x0000000004007385 */
                                                                                      /* 0x000fe20000108900 */
        /*0040*/                   CCTLL.IVALL ;                                      /* 0x00000000ff007990 */
                                                                                      /* 0x000fe20002000000 */
        /*0050*/                   CCTL.IVALL ;                                       /* 0x00000000ff00798f */
                                                                                      /* 0x000fe20002000000 */
        /*0060*/                   JMP 0x7fac32f78800 ;                               /* 0x32f788000000794a */
                                                                                      /* 0x000fea0003807fac */
        /*0070*/                   BPT.DRAIN;                                         /* 0x000000000000795c */
                                                                                      /* 0x000fc00000500000 */
        /*0080*/                   BPT.PAUSE_QUIET;                                   /* 0x000000000000795c */
                                                                                      /* 0x000fc00000600000 */
        /*0090*/                   CCTL.C.IVALL ;                                     /* 0x00000000ff00798f */
                                                                                      /* 0x000ff00002008000 */
        /*00a0*/                   CCTL.I.IVALL ;                                     /* 0x00000000ff00798f */
                                                                                      /* 0x000ff6000200c000 */
        /*00b0*/                   CCTLL.IVALL ;                                      /* 0x00000000ff007990 */
                                                                                      /* 0x000fe20002000000 */
        /*00c0*/                   CCTL.IVALL ;                                       /* 0x00000000ff00798f */
                                                                                      /* 0x000fe20002000000 */
        /*00d0*/                   JMP 0x7fac32f79200 ;                               /* 0x32f792000000794a */
                                                                                      /* 0x000fea0003807fac */
        /*00e0*/                   RET.ABS R12 0x20;                                  /* 0x000000200c007950 */
                                                                                      /* 0x000fc00003a00000 */
        /*00f0*/                   BRA 0xf0;                                          /* 0xfffffff000007947 */
                                                                                      /* 0x000fc0000383ffff */
        /*0100*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0110*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0120*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0130*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0140*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0150*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0160*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0170*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0180*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0190*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01a0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01b0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01c0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01d0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01e0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01f0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f70500
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   LDL.NA R2, [R1+0x8] ;                              /* 0x0000080001027983 */
                                                                                      /* 0x000fe20000500800 */
        /*0010*/                   MOV R4, 0x1e000000 ;                               /* 0x1e00000000047802 */
                                                                                      /* 0x000fe40000000f00 */
        /*0020*/                   MOV R5, 0x7fac ;                                   /* 0x00007fac00057802 */
                                                                                      /* 0x000fe20000000f00 */
        /*0030*/                   S2R R0, SR_VIRTUALSMID ;                           /* 0x0000000000007919 */
                                                                                      /* 0x000e220000004300 */
        /*0040*/                   MOV R6, 0x74a20 ;                                  /* 0x00074a2000067802 */
                                                                                      /* 0x000fec0000000f00 */
        /*0050*/                   IMAD.WIDE.U32 R4, R0, R6, R4 ;                     /* 0x0000000600047225 */
                                                                                      /* 0x001fea00078e0004 */
        /*0060*/                   IADD3 R4, P0, R4, 0x8, RZ ;                        /* 0x0000000804047810 */
                                                                                      /* 0x000fea0007f1e0ff */
        /*0070*/                   IADD3.X R5, R5, RZ, RZ, P0, !PT ;                  /* 0x000000ff05057210 */
                                                                                      /* 0x000fe200007fe4ff */
        /*0080*/                   S2R R0, SR_VIRTID ;                                /* 0x0000000000007919 */
                                                                                      /* 0x000e240000000300 */
        /*0090*/                   SHF.R.U32.HI R0, RZ, 0x8, R0 ;                     /* 0x00000008ff007819 */
                                                                                      /* 0x001fec0000011600 */
        /*00a0*/                   SGXT.U32 R0, R0, 0x7 ;                             /* 0x000000070000781a */
                                                                                      /* 0x000fec0000000000 */
        /*00b0*/                   ISETP.GE.U32.AND P0, PT, R0, 0x20, PT ;            /* 0x000000200000780c */
                                                                                      /* 0x000fda0003f06070 */
        /*00c0*/               @P0 IADD3 R4, P1, R4, 0x4, RZ ;                        /* 0x0000000404040810 */
                                                                                      /* 0x000fea0007f3e0ff */
        /*00d0*/               @P0 IADD3.X R5, R5, RZ, RZ, P1, !PT ;                  /* 0x000000ff05050210 */
                                                                                      /* 0x000fe40000ffe4ff */
        /*00e0*/               @P0 IADD3 R0, R0, -0x20, RZ ;                          /* 0xffffffe000000810 */
                                                                                      /* 0x000fec0007ffe0ff */
        /*00f0*/                   SHF.L.U32.HI R0, RZ, R0, 0x1 ;                     /* 0x00000001ff007419 */
                                                                                      /* 0x000fec0000010600 */
        /*0100*/                   LOP3.LUT R0, RZ, R0, RZ, 0x33, !PT ;               /* 0x00000000ff007212 */
                                                                                      /* 0x000fea00078e33ff */
        /*0110*/                   RED.E.AND.STRONG.SM.PRIVATE [R4], R0 ;             /* 0x000000000400798e */
                                                                                      /* 0x000fe20002908100 */
        /*0120*/                   RET.ABS R12 0x20;                                  /* 0x000000200c007950 */
                                                                                      /* 0x000fc00003a00000 */
        /*0130*/                   BRA 0x130;                                         /* 0xfffffff000007947 */
                                                                                      /* 0x000fc0000383ffff */
        /*0140*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0150*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0160*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0170*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0180*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0190*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01a0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01b0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01c0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01d0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01e0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01f0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f71000
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   IADD3 R1, R1, -0x8, RZ ;                           /* 0xfffffff801017810 */
                                                                                      /* 0x000fea0007ffe0ff */
        /*0010*/                   STL.64 [R1], R12 ;                                 /* 0x0000000c01007387 */
                                                                                      /* 0x0001e20000100a00 */
        /*0020*/                   MOV R2, R3 ;                                       /* 0x0000000300027202 */
                                                                                      /* 0x000fe40000000f00 */
        /*0030*/                   LDL R3, [0xffffc4] ;                               /* 0xffffc400ff037983 */
                                                                                      /* 0x000ea20000100800 */
        /*0040*/                   LDC R4, c[0x0][0x28] ;                             /* 0x00000a00ff047b82 */
                                                                                      /* 0x000e640000000800 */
        /*0050*/                   ISETP.GT.U32.AND P0, PT, R3, R4, PT ;              /* 0x000000040300720c */
                                                                                      /* 0x006fda0003f04070 */
        /*0060*/               @P0 LDL R4, [0xfffdc4] ;                               /* 0xfffdc400ff040983 */
                                                                                      /* 0x000e620000100800 */
        /*0070*/                   STL [0xfffdc4], R3 ;                               /* 0xfffdc403ff007387 */
                                                                                      /* 0x000be20000100800 */
        /*0080*/               @P0 STL [0xffffc4], R4 ;                               /* 0xffffc404ff000387 */
                                                                                      /* 0x0025e20000100800 */
        /*0090*/                   LEPC R12 ;                                         /* 0x00000000000c734e */
                                                                                      /* 0x001fe20000000000 */
        /*00a0*/                   CALL.ABS 0x7fac32f5f200;                           /* 0x32f5f20000007943 */
                                                                                      /* 0x000fc00003807fac */
        /*00b0*/                   LDL R3, [0xffffc4] ;                               /* 0xffffc400ff037983 */
                                                                                      /* 0x020ee20000100800 */
        /*00c0*/                   LDL R4, [0xfffdc4] ;                               /* 0xfffdc400ff047983 */
                                                                                      /* 0x004e640000100800 */
        /*00d0*/                   ISETP.EQ.U32.AND P0, PT, R3, R4, PT ;              /* 0x000000040300720c */
                                                                                      /* 0x00afda0003f02070 */
        /*00e0*/              @!P0 STL [0xfffdc4], R3 ;                               /* 0xfffdc403ff008387 */
                                                                                      /* 0x000fe20000100800 */
        /*00f0*/              @!P0 STL [0xffffc4], R4 ;                               /* 0xffffc404ff008387 */
                                                                                      /* 0x000fe20000100800 */
        /*0100*/                   LDL.64 R12, [R1] ;                                 /* 0x00000000010c7983 */
                                                                                      /* 0x0010640000100a00 */
        /*0110*/                   IADD3 R1, R1, 0x8, RZ ;                            /* 0x0000000801017810 */
                                                                                      /* 0x001fe20007ffe0ff */
        /*0120*/                   RET.ABS R12 0x20 ;                                 /* 0x000000200c007950 */
                                                                                      /* 0x002fc00003a00000 */
        /*0130*/                   BRA 0x130;                                         /* 0xfffffff000007947 */
                                                                                      /* 0x000fc0000383ffff */
        /*0140*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0150*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0160*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0170*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0180*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*0190*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01a0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01b0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01c0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01d0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01e0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
        /*01f0*/                   NOP;                                               /* 0x0000000000007918 */
                                                                                      /* 0x000fc00000000000 */
		..........


	code for sm_86
                Entry Point : 0x7fac32f5f200
	.headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        /*0000*/                   BRA.U 0x9d80 ;                                /* 0x00009d7100007947 */
                                                                                 /* 0x000fea0003800000 */
        /*0010*/                   NOP ;                                         /* 0x0000000000007918 */
                                                                                 /* 0x000fe20000000000 */
        /*0020*/                   NOP ;                                         /* 0x0000000000007918 */
                                                                                 /* 0x000fe20000000000 */
        /*0030*/                   NOP ;                                         /* 0x0000000000007918 */
                                                                                 /* 0x000fe20000000000 */
        /*0040*/                   IADD3 R1, R1, -0x8, RZ ;                      /* 0xfffffff801017810 */
                                                                                 /* 0x000fea0007ffe0ff */
        /*0050*/                   STL.64 [R1], R12 ;                            /* 0x0000000c01007387 */
                                                                                 /* 0x0001e20000100a00 */
        /*0060*/                   STL [0xffffa0], R1 ;                          /* 0xffffa001ff007387 */
                                                                                 /* 0x0001e20000100800 */
        /*0070*/              @!P2 BRA.U 0x9c0 ;                                 /* 0x000009410000a947 */
                                                                                 /* 0x000fea0003800000 */
        /*0080*/                   S2R R0, SR_VIRTID ;                           /* 0x0000000000007919 */
                                                                                 /* 0x000ea20000000300 */
        /*0090*/                   S2R R1, SR_VIRTUALSMID ;                      /* 0x0000000000017919 */
                                                                                 /* 0x001e620000004300 */
        /*00a0*/                   SHF.R.U32.HI R0, RZ, 0x8, R0 ;                /* 0x00000008ff007819 */
                                                                                 /* 0x004fec0000011600 */
        /*00b0*/                   SGXT.U32 R0, R0, 0x7 ;                        /* 0x000000070000781a */
                                                                                 /* 0x000fec0000000000 */
        /*00c0*/                   IMAD R2, R1, 0x40, R0 ;                       /* 0x0000004001027824 */
                                                                                 /* 0x002fe200078e0200 */
        /*00d0*/                   MOV R0, c[0x1][0x0] ;                         /* 0x0040000000007a02 */
                                                                                 /* 0x000fe40000000f00 */
        /*00e0*/                   MOV R1, c[0x1][0x4] ;                         /* 0x0040010000017a02 */
                                                                                 /* 0x001fec0000000f00 */
        /*00f0*/                   IMAD.WIDE R0, R2, c[0x1][0x8], R0 ;           /* 0x0040020002007a25 */
                                                                                 /* 0x001fea00078e0200 */
        /*0100*/                   LD.E.64 R0, [R0+0x8] ;                        /* 0x0000000800007980 */
                                                                                 /* 0x0010620000100b00 */
        /*0110*/                   S2R R2, SR_LANEID ;                           /* 0x0000000000027919 */
                                                                                 /* 0x000ea20000000000 */
        /*0120*/                   IADD3 R0, P0, R0, 0x6600, RZ ;                /* 0x0000660000007810 */
                                                                                 /* 0x003fec0007f1e0ff */
        /*0130*/                   IMAD R0, R2, 0x10, R0 ;                       /* 0x0000001002007824 */
                                                                                 /* 0x004fe200078e0200 */
        /*0140*/                   IADD3.X R1, R1, 0x0, RZ, P0, !PT ;            /* 0x0000000001017810 */
                                                                                 /* 0x001fe600007fe4ff */
        /*0150*/                   LDL.128 R4, [0xffffc0] ;                      /* 0xffffc000ff047983 */
                                                                                 /* 0x000ea20000100c00 */
        /*0160*/                   LDL.128 R8, [0xffffd0] ;                      /* 0xffffd000ff087983 */
                                                                                 /* 0x000e620000100c00 */
        /*0170*/                   ST.E.128 [R0+0x80], R4 ;                      /* 0x0000008000007385 */
                                                                                 /* 0x0041e20000100d04 */
        /*0180*/                   ST.E.128 [R0+0x280], R8 ;                     /* 0x0000028000007385 */
                                                                                 /* 0x0021e40000100d08 */
        /*0190*/                   LDL.128 R4, [0xffffe0] ;                      /* 0xffffe000ff047983 */
                                                                                 /* 0x001e640000100c00 */
        /*01a0*/                   ST.E.128 [R0+0x480], R4 ;                     /* 0x0000048000007385 */
                                                                                 /* 0x0021e40000100d04 */
        /*01b0*/                   LDL.64 R12, [0xfffff0] ;                      /* 0xfffff000ff0c7983 */
                                                                                 /* 0x001e640000100a00 */
        /*01c0*/                   ST.E.128 [R0+0x680], R12 ;                    /* 0x0000068000007385 */
                                                                                 /* 0x0021e20000100d0c */
        /*01d0*/                   LDL R3, [0xfffff8] ;                          /* 0xfffff800ff037983 */
                                                                                 /* 0x000e620000100800 */
        /*01e0*/                   IMAD R0, R2, -0xc, R0 ;                       /* 0xfffffff402007824 */
                                                                                 /* 0x001fea00078e0200 */
        /*01f0*/                   ST.E [R0], R3 ;                               /* 0x0000000000007385 */
                                                                                 /* 0x0021e20000100903 */
        /*0200*/                   ISETP.NE.AND P6, PT, RZ, 0x0, PT ;            /* 0x00000000ff00780c */
                                                                                 /* 0x000fe20003fc5270 */
        /*0210*/                   S2R R0, SR_VIRTID ;                           /* 0x0000000000007919 */
                                                                                 /* 0x001e620000000300 */
        /*0220*/                   S2R R1, SR_VIRTUALSMID ;                      /* 0x0000000000017919 */
                                                                                 /* 0x001ea20000004300 */
        /*0230*/                   SHF.R.U32.HI R0, RZ, 0x8, R0 ;                /* 0x00000008ff007819 */
                                                                                 /* 0x003fec0000011600 */
        /*0240*/                   SGXT.U32 R0, R0, 0x7 ;                        /* 0x000000070000781a */
                                                                                 /* 0x001fec0000000000 */
        /*0250*/                   IMAD R2, R1, 0x40, R0 ;                       /* 0x0000004001027824 */
                                                                                 /* 0x004fe200078e0200 */
        /*0260*/                   MOV R0, c[0x1][0x0] ;                         /* 0x0040000000007a02 */
                                                                                 /* 0x001fe40000000f00 */
        /*0270*/                   MOV R1, c[0x1][0x4] ;                         /* 0x0040010000017a02 */
                                                                                 /* 0x001fec0000000f00 */
        /*0280*/                   IMAD.WIDE R0, R2, c[0x1][0x8], R0 ;           /* 0x0040020002007a25 */
                                                                                 /* 0x001fea00078e0200 */
        /*0290*/                   LD.E.64 R0, [R0+0x8] ;                        /* 0x0000000800007980 */
                                                                                 /* 0x0010a40000100b00 */
        /*02a0*/                   S2R R3, SR_LANEID ;                           /* 0x0000000000037919 */
                                                                                 /* 0x001e640000000000 */
        /*02b0*/                   IMAD.SHL R3, R3, 0x8, RZ ;                    /* 0x0000000803037824 */
                                                                                 /* 0x003fea00078e02ff */
        /*02c0*/                   IADD3 R0, P0, R0, R3, RZ ;                    /* 0x0000000300007210 */
                                                                                 /* 0x005fea0007f1e0ff */
        /*02d0*/                   IADD3.X R1, R1, RZ, RZ, P0, !PT ;             /* 0x000000ff01017210 */
                                                                                 /* 0x001fe600007fe4ff */
        /*02e0*/                   LDS.64 R4, [R3] ;                             /* 0x0000000003047984 */
                                                                                 /* 0x0010640000000a00 */
        /*02f0*/                   ST.E.64 [R0+0x6e80], R4 ;                     /* 0x00006e8000007385 */
                                                                                 /* 0x0021e20000100b04 */
        /*0300*/                   PLOP3.LUT P0, PT, PT, PT, PT, 0x2, 0x20 ;     /* 0x000000000020781c */
                                                                                 /* 0x000fe20003f0e072 */
        /*0310*/                   S2R R0, SR_VIRTID ;                           /* 0x0000000000007919 */
                                                                                 /* 0x001e620000000300 */
        /*0320*/                   S2R R1, SR_VIRTUALSMID ;                      /* 0x0000000000017919 */
                                                                                 /* 0x001ea20000004300 */
        /*0330*/                   SHF.R.U32.HI R0, RZ, 0x8, R0 ;                /* 0x00000008ff007819 */
                                                                                 /* 0x003fec0000011600 */
        /*0340*/                   SGXT.U32 R0, R0, 0x7 ;                        /* 0x000000070000781a */
                                                                                 /* 0x001fec0000000000 */
        /*0350*/                   IMAD R8, R1, 0x40, R0 ;                       /* 0x0000004001087824 */
                                                                                 /* 0x005fe200078e0200 */
        /*0360*/                   MOV R0, c[0x1][0x0] ;                         /* 0x0040000000007a02 */
                                                                                 /* 0x001fe40000000f00 */
        /*0370*/                   MOV R1, c[0x1][0x4] ;                         /* 0x0040010000017a02 */
                                                                                 /* 0x001fec0000000f00 */
        /*0380*/                   IMAD.WIDE R0, R8, c[0x1][0x8], R0 ;           /* 0x0040020008007a25 */
                                                                                 /* 0x001fe200078e0200 */
        /*0390*/                   MOV R2, c[0x1][0x0] ;                         /* 0x0040000000027a02 */
                                                                                 /* 0x000fe40000000f00 */
        /*03a0*/                   MOV R3, c[0x1][0x4] ;                         /* 0x0040010000037a02 */
                                                                                 /* 0x001fe60000000f00 */
        /*03b0*/               @P2 LD.E.128 R8, [R0] ;                           /* 0x0000000000082980 */
                                                                                 /* 0x0010a40000100d00 */
        /*03c0*/               @P2 LD.E.128 R12, [R0+0x10] ;                     /* 0x00000010000c2980 */
                                                                                 /* 0x0010e40000100d00 */
        /*03d0*/               @P2 LD.E R4, [R2+-0x10] ;                         /* 0xfffffff002042980 */
                                                                                 /* 0x0010640000100900 */
        /*03e0*/                   S2R R6, SR_REGALLOC ;                         /* 0x0000000000067919 */
                                                                                 /* 0x001ee40000003d00 */
        /*03f0*/                   IADD3 R6, R6, -0x2, RZ ;                      /* 0xfffffffe06067810 */
                                                                                 /* 0x009fea0007ffe0ff */
        /*0400*/                   S2R R2, SR_LMEMLOSZ ;                         /* 0x0000000000027919 */
                                                                                 /* 0x001f220000003600 */
        /*0410*/                   LEA R6, R6, 0x27f, 0x7 ;                      /* 0x0000027f06067811 */
                                                                                 /* 0x001fe800078e38ff */
        /*0420*/                   MOV R7, c[0x1][0x10] ;                        /* 0x0040040000077a02 */
                                                                                 /* 0x001fe20000000f00 */
        /*0430*/                   S2R R0, SR_NTID ;                             /* 0x0000000000007919 */
                                                                                 /* 0x001ee20000002800 */
        /*0440*/                   LOP3.LUT R3, R6, 0xfffffe00, RZ, 0xc0, !PT ;  /* 0xfffffe0006037812 */
                                                                                 /* 0x001fe400078ec0ff */
        /*0450*/                   PRMT R0, R0, 0x1010, RZ ;                     /* 0x0000101000007816 */
                                                                                 /* 0x009fe400000000ff */
        /*0460*/                   LEA R6, -R2, R7, 0x5 ;                        /* 0x0000000702067211 */
                                                                                 /* 0x011fe800078e29ff */
        /*0470*/                   IADD3 R0, R0, 0x3, RZ ;                       /* 0x0000000300007810 */
                                                                                 /* 0x001fe40007ffe0ff */
        /*0480*/                   IADD3 R2, R3, R6, RZ ;                        /* 0x0000000603027210 */
                                                                                 /* 0x001fe20007ffe0ff */
        /*0490*/                   STS [0x24], RZ ;                              /* 0x000024ffff007388 */
                                                                                 /* 0x000fe20000000800 */
        /*04a0*/               @P2 STS.128 [0x30], RZ ;                          /* 0x000030ffff002388 */
                                                                                 /* 0x000fe20000000c00 */
        /*04b0*/               @P2 STS.128 [0x40], RZ ;                          /* 0x000040ffff002388 */
                                                                                 /* 0x000fe20000000c00 */
        /*04c0*/                   CCTL.E.PF1 [R10] ;                            /* 0x000000000a00798f */
                                                                                 /* 0x0041e40000000100 */
        /*04d0*/                   LOP3.LUT R9, R9, R4, RZ, 0xfc, !PT ;          /* 0x0000000409097212 */
                                                                                 /* 0x003fe400078efcff */
        /*04e0*/                   LOP3.LUT R0, R0, 0x1f001c, RZ, 0xc0, !PT ;    /* 0x001f001c00007812 */
                                                                                 /* 0x001fea00078ec0ff */
        /*04f0*/               @P2 STS.128 [0x20], R0 ;                          /* 0x00002000ff002388 */
                                                                                 /* 0x0001e20000000c00 */
        /*0500*/               @P2 STS.128 [RZ], R8 ;                            /* 0x00000008ff002388 */
                                                                                 /* 0x0001e20000000c00 */
        /*0510*/               @P2 STS.128 [0x10], R12 ;                         /* 0x0000100cff002388 */
                                                                                 /* 0x0081e40000000c00 */
        /*0520*/                   S2R R1, SR_TID.Z ;                            /* 0x0000000000017919 */
                                                                                 /* 0x001e620000002300 */
        /*0530*/                   S2R R2, SR_TID.Y ;                            /* 0x0000000000027919 */
                                                                                 /* 0x001ea40000002200 */
        /*0540*/                   IMAD R2, R1, c[0x0][0x4], R2 ;                /* 0x0000010001027a24 */
                                                                                 /* 0x007fe800078e0202 */
        /*0550*/                   S2R R1, SR_TID.X ;                            /* 0x0000000000017919 */
                                                                                 /* 0x001e640000002100 */
        /*0560*/                   IMAD R2, R2, c[0x0][0x0], R1 ;                /* 0x0000000002027a24 */
                                                                                 /* 0x003fea00078e0201 */
        /*0570*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x00000005ff027819 */
                                                                                 /* 0x001fea0000011602 */
        /*0580*/                   SHFL.IDX PT, R2, R2, 0x0, 0x0 ;               /* 0x0000000002027f89 */
                                                                                 /* 0x0010a200000e0000 */
        /*0590*/                   BAR.SYNCALL ;                                 /* 0x0000000000007b1d */
                                                                                 /* 0x000fec0000008000 */
        /*05a0*/                   LDS.64 R0, [0x8] ;                            /* 0x00000800ff007984 */
                                                                                 /* 0x001e640000000a00 */
        /*05b0*/                   LEA R0, R2, R0, 0x3 ;                         /* 0x0000000002007211 */
                                                                                 /* 0x007fea00078e18ff */
        /*05c0*/                   LD.E.NA.64.STRONG.GPU R0, [R0] ;              /* 0x0000000000007980 */
                                                                                 /* 0x001064000050eb00 */
        /*05d0*/                   SHF.L.U32 R2, R2, 0x1, RZ ;                   /* 0x0000000102027819 */
                                                                                 /* 0x001fe400000006ff */
        /*05e0*/                   PRMT R0, R0, 0x4321, R1 ;                     /* 0x0000432100007816 */
                                                                                 /* 0x003fe40000000001 */
        /*05f0*/                   SHF.R.S32.HI R1, RZ, 0x8, R1 ;                /* 0x00000008ff017819 */
                                                                                 /* 0x001fea0000011401 */
        /*0600*/                   STS.U16 [R2+0x40], R1 ;                       /* 0x0000400102007388 */
                                                                                 /* 0x0001e40000000400 */
        /*0610*/                   SHF.L.U32 R2, R2, 0x1, RZ ;                   /* 0x0000000102027819 */
                                                                                 /* 0x001fea00000006ff */
        /*0620*/                   STS [R2+0x80], R0 ;                           /* 0x0000800002007388 */
                                                                                 /* 0x0001e20000000800 */
        /*0630*/                   BAR.SYNCALL ;                                 /* 0x0000000000007b1d */
                                                                                 /* 0x000fec0000008000 */
        /*0640*/                   S2R R3, SR_TID.Z ;                            /* 0x0000000000037919 */
                                                                                 /* 0x001e620000002300 */
        /*0650*/                   S2R R2, SR_TID.Y ;                            /* 0x0000000000027919 */
                                                                                 /* 0x001ea40000002200 */
        /*0660*/                   IMAD R2, R3, c[0x0][0x4], R2 ;                /* 0x0000010003027a24 */
                                                                                 /* 0x007fe800078e0202 */
        /*0670*/                   S2R R3, SR_TID.X ;                            /* 0x0000000000037919 */
                                                                                 /* 0x001e640000002100 */
        /*0680*/                   IMAD R2, R2, c[0x0][0x0], R3 ;                /* 0x0000000002027a24 */
                                                                                 /* 0x003fea00078e0203 */
        /*0690*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x00000005ff027819 */
                                                                                 /* 0x001fea0000011602 */
        /*06a0*/                   SHFL.IDX PT, R2, R2, 0x0, 0x0 ;               /* 0x0000000002027f89 */
                                                                                 /* 0x00116400000e0000 */
        /*06b0*/                   LDS.64 R0, [0x8] ;                            /* 0x00000800ff007984 */
                                                                                 /* 0x001e620000000a00 */
        /*06c0*/                   S2R R3, SR_EQMASK ;                           /* 0x0000000000037919 */
                                                                                 /* 0x001ea20000003800 */
        /*06d0*/                   S2R R9, SR_LANEID ;                           /* 0x0000000000097919 */
                                                                                 /* 0x001ea20000000000 */
        /*06e0*/                   P2R R8, PR, RZ, 0xff ;                        /* 0x000000ffff087803 */
                                                                                 /* 0x001fe40000000000 */
        /*06f0*/                   LEA R0, R9, R0, 0x4 ;                         /* 0x0000000009007211 */
                                                                                 /* 0x007fe400078e20ff */
        /*0700*/                   R2P PR, R3, 0xf ;                             /* 0x0000000f03007804 */
                                                                                 /* 0x004fda0000000000 */
        /*0710*/               @P0 B2R R4, 0x0 ;                                 /* 0x000000000004031c */
                                                                                 /* 0x001e620000000000 */
        /*0720*/               @P0 B2R R5, 0x1 ;                                 /* 0x004000000005031c */
                                                                                 /* 0x001ee20000000000 */
        /*0730*/               @P0 B2R R6, 0x2 ;                                 /* 0x008000000006031c */
                                                                                 /* 0x001f220000000000 */
        /*0740*/               @P0 B2R R7, 0x3 ;                                 /* 0x00c000000007031c */
                                                                                 /* 0x001e240000000000 */
        /*0750*/                   IMAD.WIDE R2, R2, 0x2e0, R0 ;                 /* 0x000002e002027825 */
                                                                                 /* 0x021fe200078e0200 */
        /*0760*/               @P1 B2R R4, 0x4 ;                                 /* 0x010000000004131c */
                                                                                 /* 0x003ea20000000000 */
        /*0770*/               @P1 B2R R5, 0x5 ;                                 /* 0x014000000005131c */
                                                                                 /* 0x009ee20000000000 */
        /*0780*/               @P1 B2R R6, 0x6 ;                                 /* 0x018000000006131c */
                                                                                 /* 0x011e620000000000 */
        /*0790*/               @P1 B2R R7, 0x7 ;                                 /* 0x01c000000007131c */
                                                                                 /* 0x001f620000000000 */
        /*07a0*/               @P2 B2R R4, 0x8 ;                                 /* 0x020000000004231c */
                                                                                 /* 0x005f220000000000 */
        /*07b0*/               @P2 B2R R5, 0x9 ;                                 /* 0x024000000005231c */
                                                                                 /* 0x009ee20000000000 */
        /*07c0*/               @P2 B2R R6, 0xa ;                                 /* 0x028000000006231c */
                                                                                 /* 0x003e620000000000 */
        /*07d0*/               @P2 B2R R7, 0xb ;                                 /* 0x02c000000007231c */
                                                                                 /* 0x021ea20000000000 */
        /*07e0*/                   ISETP.LT.AND P6, PT, R9, 0x4, PT ;            /* 0x000000040900780c */
                                                                                 /* 0x000ff00003fc1270 */
        /*07f0*/               @P3 B2R R4, 0xc ;                                 /* 0x030000000004331c */
                                                                                 /* 0x011f220000000000 */
        /*0800*/               @P3 B2R R5, 0xd ;                                 /* 0x034000000005331c */
                                                                                 /* 0x009f220000000000 */
        /*0810*/               @P3 B2R R6, 0xe ;                                 /* 0x038000000006331c */
                                                                                 /* 0x003f220000000000 */
        /*0820*/               @P3 B2R R7, 0xf ;                                 /* 0x03c000000007331c */
                                                                                 /* 0x005f220000000000 */
        /*0830*/                   B2R.WARP R10 ;                                /* 0x00000000000a731c */
                                                                                 /* 0x001ea20000008000 */
        /*0840*/               @P6 ST.E.128 [R0+0x5e10], R4 ;                    /* 0x00005e1000006385 */
                                                                                 /* 0x0101e20000100d04 */
        /*0850*/                   R2P PR, R8 ;                                  /* 0x000000ff08007804 */
                                                                                 /* 0x000fe20000000000 */
        /*0860*/                   R2B 0xf, RZ ;                                 /* 0x03c000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0870*/                   R2B 0xe, RZ ;                                 /* 0x038000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0880*/                   R2B 0xd, RZ ;                                 /* 0x034000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0890*/                   R2B 0xc, RZ ;                                 /* 0x030000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*08a0*/                   R2B 0xb, RZ ;                                 /* 0x02c000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*08b0*/                   R2B 0xa, RZ ;                                 /* 0x028000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*08c0*/                   R2B 0x9, RZ ;                                 /* 0x024000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*08d0*/                   R2B 0x8, RZ ;                                 /* 0x020000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*08e0*/                   R2B 0x7, RZ ;                                 /* 0x01c000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*08f0*/                   R2B 0x6, RZ ;                                 /* 0x018000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0900*/                   R2B 0x5, RZ ;                                 /* 0x014000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0910*/                   R2B 0x4, RZ ;                                 /* 0x010000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0920*/                   R2B 0x3, RZ ;                                 /* 0x00c000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0930*/                   R2B 0x2, RZ ;                                 /* 0x008000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0940*/                   R2B 0x1, RZ ;                                 /* 0x004000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0950*/                   R2B 0x0, RZ ;                                 /* 0x000000ff0000731e */
                                                                                 /* 0x000fe20000000000 */
        /*0960*/                   ISETP.EQ.AND P6, PT, R9, 0x0, PT ;            /* 0x000000000900780c */
                                                                                 /* 0x000fd80003fc2270 */
        /*0970*/                   R2B.WARP 0x0, RZ ;                            /* 0x000000ff0000731e */
                                                                                 /* 0x000fe20000008000 */
        /*0980*/               @P6 ST.E [R2+0x21c], R10 ;                        /* 0x0000021c02006385 */
                                                                                 /* 0x0041e2000010090a */
        /*0990*/                   BRA.U 0xd50 ;                                 /* 0x000003b100007947 */
                                                                                 /* 0x000fea0003800000 */
        /*09a0*/                   NOP ;                                         /* 0x0000000000007918 */
                                                                                 /* 0x000fe20000000000 */
        /*09b0*/                   NOP ;                                         /* 0x0000000000007918 */
                                                                                 /* 0x000fe20000000000 */
        /*09c0*/                   S2R R1, SR_TID.Z ;                            /* 0x0000000000017919 */
                                                                                 /* 0x001e620000002300 */
        /*09d0*/                   S2R R2, SR_TID.Y ;                            /* 0x0000000000027919 */
                                                                                 /* 0x000ea40000002200 */
        /*09e0*/                   IMAD R2, R1, c[0x0][0x4], R2 ;                /* 0x0000010001027a24 */
                                                                                 /* 0x006fe800078e0202 */
        /*09f0*/                   S2R R1, SR_TID.X ;                            /* 0x0000000000017919 */
                                                                                 /* 0x000e640000002100 */
        /*0a00*/                   IMAD R2, R2, c[0x0][0x0], R1 ;                /* 0x0000000002027a24 */
                                                                                 /* 0x002fea00078e0201 */
        /*0a10*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x00000005ff027819 */
                                                                                 /* 0x000fea0000011602 */
        /*0a20*/                   SHFL.IDX PT, R2, R2, 0x0, 0x0 ;               /* 0x0000000002027f89 */
                                                                                 /* 0x000e6200000e0000 */
        /*0a30*/                   BAR.SYNCALL ;                                 /* 0x0000000000007b1d */
                                                                                 /* 0x000fec0000008000 */
        /*0a40*/                   LDS.64 R0, [0x8] ;                            /* 0x00000800ff007984 */
                                                                                 /* 0x000ea40000000a00 */
        /*0a50*/                   LEA R0, R2, R0, 0x3 ;                         /* 0x0000000002007211 */
                                                                                 /* 0x006fea00078e18ff */
        /*0a60*/                   LD.E.NA.64.STRONG.GPU R0, [R0] ;              /* 0x0000000000007980 */
                                                                                 /* 0x000462000050eb00 */
        /*0a70*/                   SHF.L.U32 R2, R2, 0x1, RZ ;                   /* 0x0000000102027819 */
                                                                                 /* 0x000fe400000006ff */
        /*0a80*/                   PRMT R0, R0, 0x4321, R1 ;                     /* 0x0000432100007816 */
                                                                                 /* 0x006fe40000000001 */
        /*0a90*/                   SHF.R.S32.HI R1, RZ, 0x8, R1 ;                /* 0x00000008ff017819 */
                                                                                 /* 0x000fea0000011401 */
        /*0aa0*/                   STS.U16 [R2+0x40], R1 ;                       /* 0x0000400102007388 */
                                                                                 /* 0x0001e40000000400 */
        /*0ab0*/                   SHF.L.U32 R2, R2, 0x1, RZ ;                   /* 0x0000000102027819 */
                                                                                 /* 0x001fea00000006ff */
        /*0ac0*/                   STS [R2+0x80], R0 ;                           /* 0x0000800002007388 */
                                                                                 /* 0x0001e20000000800 */
        /*0ad0*/                   BAR.SYNCALL ;                                 /* 0x0000000000007b1d */
                                                                                 /* 0x000fec0000008000 */
        /*0ae0*/                   S2R R0, SR_TID.Z ;                            /* 0x0000000000007919 */
                                                                                 /* 0x001ea20000002300 */
        /*0af0*/                   S2R R2, SR_TID.Y ;                            /* 0x0000000000027919 */
                                                                                 /* 0x001e640000002200 */
        /*0b00*/                   IMAD R2, R0, c[0x0][0x4], R2 ;                /* 0x0000010000027a24 */
                                                                                 /* 0x007fe800078e0202 */
        /*0b10*/                   S2R R0, SR_TID.X ;                            /* 0x0000000000007919 */
                                                                                 /* 0x001e640000002100 */
        /*0b20*/                   IMAD R2, R2, c[0x0][0x0], R0 ;                /* 0x0000000002027a24 */
                                                                                 /* 0x003fea00078e0200 */
        /*0b30*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x00000005ff027819 */
                                                                                 /* 0x001fea0000011602 */
        /*0b40*/                   SHFL.IDX PT, R2, R2, 0x0, 0x0 ;               /* 0x0000000002027f89 */
                                                                                 /* 0x0010a400000e0000 */
        /*0b50*/                   LDS.64 R0, [0x8] ;                            /* 0x00000800ff007984 */
                                                                                 /* 0x001e640000000a00 */
        /*0b60*/                   IMAD.WIDE R0, R2, 0x2e0, R0 ;                 /* 0x000002e002007825 */
                                                                                 /* 0x007fe600078e0200 */
        /*0b70*/                   B2R.WARP R2 ;                                 /* 0x000000000002731c */
                                                                                 /* 0x001e620000008000 */
        /*0b80*/                   R2B.WARP 0x0, RZ ;                            /* 0x000000ff0000731e */
                                                                                 /* 0x000fe20000008000 */
        /*0b90*/                   ST.E [R0+0x21c], R2 ;                         /* 0x0000021c00007385 */
                                                                                 /* 0x0021e40000100902 */
        /*0ba0*/                   S2R R0, SR_TID.Z ;                            /* 0x0000000000007919 */
                                                                                 /* 0x001ea20000002300 */
        /*0bb0*/                   S2R R2, SR_TID.Y ;                            /* 0x0000000000027919 */
                                                                                 /* 0x001e640000002200 */
        /*0bc0*/                   IMAD R2, R0, c[0x0][0x4], R2 ;                /* 0x0000010000027a24 */
                                                                                 /* 0x007fe800078e0202 */
        /*0bd0*/                   S2R R0, SR_TID.X ;                            /* 0x0000000000007919 */
                                                                                 /* 0x001e640000002100 */
        /*0be0*/                   IMAD R2, R2, c[0x0][0x0], R0 ;                /* 0x0000000002027a24 */
                                                                                 /* 0x003fea00078e0200 */
        /*0bf0*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x00000005ff027819 */
                                                                                 /* 0x001fea0000011602 */
        /*0c00*/                   SHFL.IDX PT, R2, R2, 0x0, 0x0 ;               /* 0x0000000002027f89 */
                                                                                 /* 0x00106400000e0000 */
        /*0c10*/                   SHF.L.U32 R0, R2, 0x2, RZ ;                   /* 0x0000000202007819 */
                                                                                 /* 0x003fe400000006ff */
        /*0c20*/                   SHF.L.U32 R1, R2, 0x1, RZ ;                   /* 0x0000000102017819 */
                                                                                 /* 0x001fe800000006ff */
        /*0c30*/                   LDS R0, [R0+0x80] ;                           /* 0x0000800000007984 */
                                                                                 /* 0x0010640000000800 */
        /*0c40*/                   LDS.U16 R1, [R1+0x40] ;                       /* 0x0000400001017984 */
                                                                                 /* 0x0010a40000000400 */
        /*0c50*/                   PRMT R1, R0, 0x6543, R1 ;                     /* 0x0000654300017816 */
                                                                                 /* 0x007fe60000000001 */
        /*0c60*/                   PRMT R0, RZ, 0x6540, R0 ;                     /* 0x00006540ff007816 */
                                                                                 /* 0x001fe60000000000 */
        /*0c70*/                   SGXT R1, R1, 0x11 ;                           /* 0x000000110101781a */
                                                                                 /* 0x001fe20000000200 */
        /*0c80*/                   S2R R2, SR_LANEID ;                           /* 0x0000000000027919 */
                                                                                 /* 0x001e640000000000 */
        /*0c90*/                   IMAD R0, R2, 0x10, R0 ;                       /* 0x0000001002007824 */
                                                                                 /* 0x003fe600078e0200 */
        /*0ca0*/                   LDL.128 R4, [0xffffc0] ;                      /* 0xffffc000ff047983 */
                                                                                 /* 0x000e620000100c00 */
        /*0cb0*/                   LDL.128 R8, [0xffffd0] ;                      /* 0xffffd000ff087983 */
                                                                                 /* 0x000ea20000100c00 */
        /*0cc0*/                   ST.E.128 [R0+0x80], R4 ;                      /* 0x0000008000007385 */
                                                                                 /* 0x0021e20000100d04 */
        /*0cd0*/                   ST.E.128 [R0+0x280], R8 ;                     /* 0x0000028000007385 */
                                                                                 /* 0x0041e40000100d08 */
        /*0ce0*/                   LDL.128 R4, [0xffffe0] ;                      /* 0xffffe000ff047983 */
                                                                                 /* 0x001f240000100c00 */
        /*0cf0*/                   ST.E.128 [R0+0x480], R4 ;                     /* 0x0000048000007385 */
                                                                                 /* 0x0101e40000100d04 */
        /*0d00*/                   LDL.64 R12, [0xfffff0] ;                      /* 0xfffff000ff0c7983 */
                                                                                 /* 0x001ee40000100a00 */
        /*0d10*/                   ST.E.128 [R0+0x680], R12 ;                    /* 0x0000068000007385 */
                                                                                 /* 0x0081e20000100d0c */
        /*0d20*/                   LDL R3, [0xfffff8] ;                          /* 0xfffff800ff037983 */
                                                                                 /* 0x000e620000100800 */
        /*0d30*/                   IMAD R0, R2, -0xc, R0 ;                       /* 0xfffffff402007824 */
                                                                                 /* 0x001fea00078e0200 */
        /*0d40*/                   ST.E [R0], R3 ;                               /* 0x0000000000007385 */
                                                                                 /* 0x0021e40000100903 */
        /*0d50*/                   S2R R3, SR_TID.Z ;                            /* 0x0000000000037919 */
                                                                                 /* 0x003ea20000002300 */
        /*0d60*/                   S2R R2, SR_TID.Y ;                            /* 0x0000000000027919 */
                                                                                 /* 0x003e640000002200 */
        /*0d70*/                   IMAD R2, R3, c[0x0][0x4], R2 ;                /* 0x0000010003027a24 */
                                                                                 /* 0x006fe800078e0202 */
        /*0d80*/                   S2R R3, SR_TID.X ;                            /* 0x0000000000037919 */
                                                                                 /* 0x000e640000002100 */
        /*0d90*/                   IMAD R2, R2, c[0x0][0x0], R3 ;                /* 0x0000000002027a24 */
                                                                                 /* 0x002fea00078e0203 */
        /*0da0*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x00000005ff027819 */
                                                                                 /* 0x000fea0000011602 */
        /*0db0*/                   SHFL.IDX PT, R2, R2, 0x0, 0x0 ;               /* 0x0000000002027f89 */
                                                                                 /* 0x0004e200000e0000 */
        /*0dc0*/                   LDS.64 R0, [0x8] ;                            /* 0x00000800ff007984 */
                                                                                 /* 0x003e640000000a00 */
        /*0dd0*/                   IMAD.WIDE R2, R2, 0x2e0, R0 ;                 /* 0x000002e002027825 */
                                                                                 /* 0x00efea00078e0200 */
        /*0de0*/                   IADD3 R2, P0, R2, 0x350, RZ ;                 /* 0x0000035002027810 */
                                                                                 /* 0x000fea0007f1e0ff */
        /*0df0*/                   IADD3.X R3, R3, RZ, RZ, P0, !PT ;             /* 0x000000ff03037210 */
                                                                                 /* 0x000fe200007fe4ff */
        /*0e00*/                   S2R R11, SR_LANEID ;                          /* 0x00000000000b7919 */
                                                                                 /* 0x005e640000000000 */
        /*0e10*/                   ISETP.EQ.AND P0, PT, R11, RZ, PT ;            /* 0x000000ff0b00720c */
                                                                                 /* 0x002fd80003f02270 */
        /*0e20*/                   IMAD.WIDE R0, R11, 0x8, R2 ;                  /* 0x000000080b007825 */
                                                                                 /* 0x000fe200078e0202 */
        /*0e30*/               @P0 LDL.64 R4, [0xffffa8] ;                       /* 0xffffa800ff040983 */
                                                                                 /* 0x011e620000100a00 */
        /*0e40*/                   LDL.64 R6, [0xffffb0] ;                       /* 0xffffb000ff067983 */
                                                                                 /* 0x011ea40000100a00 */
        /*0e50*/               @P0 LDL R8, [0xffffb8] ;                          /* 0xffffb800ff080983 */
                                                                                 /* 0x005e240000100800 */
        /*0e60*/               @P0 LDL R9, [0xfffffc] ;                          /* 0xfffffc00ff090983 */
                                                                                 /* 0x005ee20000100800 */
        /*0e70*/               @P0 ST.E.64 [R2+0x140], R4 ;                      /* 0x0000014002000385 */
                                                                                 /* 0x0023e20000100b04 */
        /*0e80*/                   ST.E.64 [R0], R6 ;                            /* 0x0000000000007385 */
                                                                                 /* 0x0043e20000100b06 */
        /*0e90*/               @P0 ST.E [R2+0x178], R8 ;                         /* 0x0000017802000385 */
                                                                                 /* 0x0013e20000100908 */
        /*0ea0*/               @P0 ST.E [R2+0x168], R9 ;                         /* 0x0000016802000385 */
                                                                                 /* 0x0083e40000100909 */
        /*0eb0*/               @P0 BMOV.32 R4, ATEXIT_PC.HI ;                    /* 0x000000001f040355 */
                                                                                 /* 0x002e220000000000 */
        /*0ec0*/               @P0 BMOV.32 R5, ATEXIT_PC.LO ;                    /* 0x000000001e050355 */
                                                                                 /* 0x002ea40000000000 */
        /*0ed0*/               @P0 ST.E.64 [R2+0x160], R4 ;                      /* 0x0000016002000385 */
                                                                                 /* 0x0053e40000100b04 */
        /*0ee0*/               @P0 BMOV.32 R4, API_CALL_DEPTH ;                  /* 0x000000001d040355 */
                                                                                 /* 0x002e220000000000 */
        /*0ef0*/               @P0 BMOV.32 R6, MEXITED ;                         /* 0x0000000018060355 */
                                                                                 /* 0x002ee20000000000 */
        /*0f00*/               @P0 BMOV.32 R7, MATEXIT ;                         /* 0x000000001b070355 */
                                                                                 /* 0x002ea20000000000 */
        /*0f10*/               @P0 ST.E [R2+0x174], R4 ;                         /* 0x0000017402000385 */
                                                                                 /* 0x0013e20000100904 */
        /*0f20*/               @P0 ST.E [R2+0x16c], R6 ;                         /* 0x0000016c02000385 */
                                                                                 /* 0x0083e20000100906 */
        /*0f30*/               @P0 ST.E [R2+0x170], R7 ;                         /* 0x0000017002000385 */
                                                                                 /* 0x0043e40000100907 */
        /*0f40*/               @P0 BMOV.32 R8, THREAD_STATE_ENUM.0 ;             /* 0x0000000010080355 */
                                                                                 /* 0x002ee20000000000 */
        /*0f50*/               @P0 BMOV.32 R9, THREAD_STATE_ENUM.1 ;             /* 0x0000000011090355 */
                                                                                 /* 0x002e240000000000 */
        /*0f60*/               @P0 BMOV.32 R10, THREAD_STATE_ENUM.2 ;            /* 0x00000000120a0355 */
                                                                                 /* 0x005f220000000000 */
        /*0f70*/               @P0 BMOV.32 R11, THREAD_STATE_ENUM.3 ;            /* 0x00000000130b0355 */
                                                                                 /* 0x000ea20000000000 */
        /*0f80*/               @P0 BMOV.32 R4, THREAD_STATE_ENUM.4 ;             /* 0x0000000014040355 */
                                                                                 /* 0x002f620000000000 */
        /*0f90*/               @P0 ST.E.128 [R2+0x180], R8 ;                     /* 0x0000018002000385 */
                                                                                 /* 0x01d3e20000100d08 */
        /*0fa0*/               @P0 ST.E [R2+0x190], R4 ;                         /* 0x0000019002000385 */
                                                                                 /* 0x0203e40000100904 */
        /*0fb0*/               @P0 LDL R4, [0xffffbc] ;                          /* 0xffffbc00ff040983 */
                                                                                 /* 0x002e220000100800 */
        /*0fc0*/               @P0 BMOV.32.CLEAR RZ, B0 ;                        /* 0x0000000000ff0355 */
                                                                                 /* 0x000fe20000100000 */
        /*0fd0*/               @P0 BMOV.32.CLEAR R5, B1 ;                        /* 0x0000000001050355 */
                                                                                 /* 0x002f220000100000 */
        /*0fe0*/               @P0 BMOV.32.CLEAR R6, B2 ;                        /* 0x0000000002060355 */
                                                                                 /* 0x002f220000100000 */
        /*0ff0*/               @P0 BMOV.32.CLEAR R7, B3 ;                        /* 0x0000000003070355 */
                                                                                 /* 0x002e640000100000 */
        /*1000*/               @P0 BMOV.32.CLEAR R8, B4 ;                        /* 0x0000000004080355 */
                                                                                 /* 0x002ea20000100000 */
        /*1010*/               @P0 BMOV.32.CLEAR R9, B5 ;                        /* 0x0000000005090355 */
                                                                                 /* 0x002f220000100000 */
        /*1020*/               @P0 BMOV.32.CLEAR R10, B6 ;                       /* 0x00000000060a0355 */
                                                                                 /* 0x002ee20000100000 */
        /*1030*/               @P0 BMOV.32.CLEAR R11, B7 ;                       /* 0x00000000070b0355 */
                                                                                 /* 0x002f620000100000 */
        /*1040*/               @P0 ST.E.128 [R2+0x100], R4 ;                     /* 0x0000010002000385 */
                                                                                 /* 0x0133e20000100d04 */
        /*1050*/               @P0 ST.E.128 [R2+0x110], R8 ;                     /* 0x0000011002000385 */
                                                                                 /* 0x03c3e40000100d08 */
        /*1060*/               @P0 BMOV.32.CLEAR R4, B8 ;                        /* 0x0000000008040355 */
                                                                                 /* 0x002e640000100000 */
        /*1070*/               @P0 BMOV.32.CLEAR R5, B9 ;                        /* 0x0000000009050355 */
                                                                                 /* 0x002e220000100000 */
        /*1080*/               @P0 BMOV.32.CLEAR R6, B10 ;                       /* 0x000000000a060355 */
                                                                                 /* 0x002e220000100000 */
        /*1090*/               @P0 BMOV.32.CLEAR R7, B11 ;                       /* 0x000000000b070355 */
                                                                                 /* 0x002e220000100000 */
        /*10a0*/               @P0 BMOV.32.CLEAR R8, B12 ;                       /* 0x000000000c080355 */
                                                                                 /* 0x002f620000100000 */
        /*10b0*/               @P0 BMOV.32.CLEAR R9, B13 ;                       /* 0x000000000d090355 */
                                                                                 /* 0x002ea20000100000 */
        /*10c0*/               @P0 BMOV.32.CLEAR R10, B14 ;                      /* 0x000000000e0a0355 */
                                                                                 /* 0x002ee20000100000 */
        /*10d0*/               @P0 BMOV.32.CLEAR R11, B15 ;                      /* 0x000000000f0b0355 */
                                                                                 /* 0x002f620000100000 */
        /*10e0*/               @P0 ST.E.128 [R2+0x120], R4 ;                     /* 0x0000012002000385 */
                                                                                 /* 0x0033e20000100d04 */
        /*10f0*/               @P0 ST.E.128 [R2+0x130], R8 ;                     /* 0x0000013002000385 */
                                                                                 /* 0x02c3e20000100d08 */
        /*1100*/                   RPCMOV.64 Rpc, URZ ;                          /* 0x0000003f00007d54 */
                                                                                 /* 0x000fec0008000000 */
        /*1110*/                   BMOV.32 TRAP_RETURN_MASK, 0xffffffff ;        /* 0xffffffff17007956 */
                                                                                 /* 0x000fe20000000000 */
        /*1120*/                   BMOV.64 ATEXIT_PC, 0x0 ;                      /* 0x0000000000007957 */
                                                                                 /* 0x000fe20000000000 */
        /*1130*/                   BMOV.32 API_CALL_DEPTH, 0x0 ;                 /* 0x000000001d007956 */
                                                                                 /* 0x000fe20000000000 */
        /*1140*/                   BMOV.32 OPT_STACK, 0x0 ;                      /* 0x000000001c007956 */
                                                                                 /* 0x000fe20000000000 */
        /*1150*/                   BMOV.32 THREAD_STATE_ENUM.0, 0xffffffff ;     /* 0xffffffff10007956 */
                                                                                 /* 0x000fe20000000000 */
        /*1160*/                   BMOV.32 THREAD_STATE_ENUM.1, 0x0 ;            /* 0x0000000011007956 */
                                                                                 /* 0x000fe20000000000 */
        /*1170*/                   BMOV.32 THREAD_STATE_ENUM.2, 0x0 ;            /* 0x0000000012007956 */
                                                                                 /* 0x000fe20000000000 */
        /*1180*/                   BMOV.32 THREAD_STATE_ENUM.3, 0x0 ;            /* 0x0000000013007956 */
                                                                                 /* 0x000fe20000000000 */
        /*1190*/                   BMOV.32 THREAD_STATE_ENUM.4, 0x0 ;            /* 0x0000000014007956 */
                                                                                 /* 0x000fe20000000000 */
        /*11a0*/                   S2R R0, SR_TID.Z ;                            /* 0x0000000000007919 */
                                                                                 /* 0x002ee20000002300 */
        /*11b0*/                   S2R R2, SR_TID.Y ;                            /* 0x0000000000027919 */
                                                                                 /* 0x002ea40000002200 */
        /*11c0*/                   IMAD R2, R0, c[0x0][0x4], R2 ;                /* 0x0000010000027a24 */
                                                                                 /* 0x00efe800078e0202 */
        /*11d0*/                   S2R R0, SR_TID.X ;                            /* 0x0000000000007919 */
                                                                                 /* 0x002ea40000002100 */
        /*11e0*/                   IMAD R2, R2, c[0x0][0x0], R0 ;                /* 0x0000000002027a24 */
                                                                                 /* 0x006fea00078e0200 */
        /*11f0*/                   SHF.R.U32.HI R2, RZ, 0x5, R2 ;                /* 0x00000005ff027819 */
                                                                                 /* 0x002fea0000011602 */