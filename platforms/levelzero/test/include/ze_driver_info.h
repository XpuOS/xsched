#pragma once

#define ZE_ASSERT(cmd)                             \
    do                                             \
    {                                              \
        ze_result_t res = cmd;                     \
        if (res != ZE_RESULT_SUCCESS)              \
        {                                          \
            printf("levelzero error 0x%x\n", res); \
            exit(1);                               \
        }                                          \
    } while (0);

void PrintZEDriverInfo();