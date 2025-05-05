#pragma once

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/utils/xassert.h"

// Helper macros to count arguments
#define COUNT_ARGS(...) COUNT_ARGS_IMPL(__VA_ARGS__ __VA_OPT__(,) 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, N, ...) N

// Helper macro to concatenate
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define CONCAT_IMPL(a, b) a##b

// Helper macros to handle function parameters and types
#define DECLARE_PARAM(type, name) type name
#define DECLARE_TYPE(type, name) type
#define DECLARE_ARG(type, name) name
#define DECLARE_PRIVATE_PARAM(type, name) type name##_
#define DECLARE_PRIVATE_ARG(type, name) name##_
#define DECLARE_COPY_PRIVATE_ARG(type, name) name##_(name)

// Save the address in the macro and call the function
#define DEFINE_ADDRESS_CALL_WITH_PREFIX(prefix, addr, ret_t, func, ...)            \
    prefix ret_t func(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) {           \
        using FuncPtr = ret_t (*)(FOR_EACH_PAIR_COMMA(DECLARE_TYPE, __VA_ARGS__)); \
        static const auto func_ptr = reinterpret_cast<FuncPtr>(addr);              \
        XDEBG("call "#func"() @ %p", func_ptr);                                    \
        return func_ptr(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__));            \
    }

// Directly call the target function
#define DEFINE_REDIRECT_CALL_WITH_PREFIX(prefix, target, ret_t, func, ...) \
    prefix ret_t func(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) {   \
        XDEBG("redirect "#func"() -> "#target"()");                        \
        return target(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__));      \
    }

// Check the ptr and directly call the target function
#define DEFINE_REDIRECT_ADDRESS_CALL_WITH_PREFIX(prefix, target_ptr, ret_t, func, ...) \
    prefix ret_t func(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) {               \
        XASSERT(target_ptr != nullptr, "target_ptr is nullptr");                       \
        XDEBG("redirect "#func"() -> %p", target_ptr);                                 \
        return target_ptr(FOR_EACH_PAIR_COMMA(DECLARE_ARG, __VA_ARGS__));              \
    }

#define DEFINE_STATIC_ADDRESS_CALL(addr, ret_t, func, ...)      \
    DEFINE_ADDRESS_CALL_WITH_PREFIX(static, addr, ret_t, func, __VA_ARGS__)

#define DEFINE_STATIC_REDIRECT_CALL(target, ret_t, func, ...)   \
    DEFINE_REDIRECT_CALL_WITH_PREFIX(static, target, ret_t, func, __VA_ARGS__)

#define DEFINE_STATIC_ADDRESS_REDIRECT_CALL(target_ptr, ret_t, func, ...)   \
    DEFINE_REDIRECT_ADDRESS_CALL_WITH_PREFIX(static, target_ptr, ret_t, func, __VA_ARGS__)

#define DEFINE_C_REDIRECT_CALL(target, ret_t, func, ...)        \
    DEFINE_REDIRECT_CALL_WITH_PREFIX(extern "C", target, ret_t, func, __VA_ARGS__)

#define DEFINE_EXPORT_C_REDIRECT_CALL(target, ret_t, func, ...) \
    DEFINE_REDIRECT_CALL_WITH_PREFIX(EXPORT_C_FUNC, target, ret_t, func, __VA_ARGS__)

#define DEFINE_EXPORT_CXX_REDIRECT_CALL(target, ret_t, func, ...) \
    DEFINE_REDIRECT_CALL_WITH_PREFIX(EXPORT_CXX_FUNC, target, ret_t, func, __VA_ARGS__)

// FOR_EACH_PAIR_COMMA implementation for handling parameter pairs
#define FOR_EACH_PAIR_COMMA(macro, ...) \
    CONCAT(FOR_EACH_PAIR_COMMA_, COUNT_ARGS(__VA_ARGS__))(macro, __VA_ARGS__)

// FOR_EACH_PAIR_SEMICOLON implementation for defining parameter pairs
#define FOR_EACH_PAIR_SEMICOLON(macro, ...) \
    CONCAT(FOR_EACH_PAIR_SEMICOLON_, COUNT_ARGS(__VA_ARGS__))(macro, __VA_ARGS__)

// Implement FOR_EACH_PAIR_COMMA_N for different argument counts (0 to 16 parameters = 0 to 32 arguments)
#define FOR_EACH_PAIR_COMMA_0(macro, ...)

#define FOR_EACH_PAIR_COMMA_2(macro, t0, n0) \
    macro(t0, n0)
#define FOR_EACH_PAIR_COMMA_4(macro, t0, n0, t1, n1) \
    macro(t0, n0), macro(t1, n1)
#define FOR_EACH_PAIR_COMMA_6(macro, t0, n0, t1, n1, t2, n2) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2)
#define FOR_EACH_PAIR_COMMA_8(macro, t0, n0, t1, n1, t2, n2, t3, n3) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3)
#define FOR_EACH_PAIR_COMMA_10(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4)
#define FOR_EACH_PAIR_COMMA_12(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5)
#define FOR_EACH_PAIR_COMMA_14(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6)
#define FOR_EACH_PAIR_COMMA_16(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7)
#define FOR_EACH_PAIR_COMMA_18(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8)
#define FOR_EACH_PAIR_COMMA_20(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9)
#define FOR_EACH_PAIR_COMMA_22(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10)
#define FOR_EACH_PAIR_COMMA_24(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11)
#define FOR_EACH_PAIR_COMMA_26(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12)
#define FOR_EACH_PAIR_COMMA_28(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12), macro(t13, n13)
#define FOR_EACH_PAIR_COMMA_30(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12), macro(t13, n13), macro(t14, n14)
#define FOR_EACH_PAIR_COMMA_32(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12), macro(t13, n13), macro(t14, n14), macro(t15, n15)
#define FOR_EACH_PAIR_COMMA_34(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12), macro(t13, n13), macro(t14, n14), macro(t15, n15), macro(t16, n16)
#define FOR_EACH_PAIR_COMMA_36(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16, t17, n17) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12), macro(t13, n13), macro(t14, n14), macro(t15, n15), macro(t16, n16), macro(t17, n17)
#define FOR_EACH_PAIR_COMMA_38(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16, t17, n17, t18, n18) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12), macro(t13, n13), macro(t14, n14), macro(t15, n15), macro(t16, n16), macro(t17, n17), macro(t18, n18)
#define FOR_EACH_PAIR_COMMA_40(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16, t17, n17, t18, n18, t19, n19) \
    macro(t0, n0), macro(t1, n1), macro(t2, n2), macro(t3, n3), macro(t4, n4), macro(t5, n5), macro(t6, n6), macro(t7, n7), macro(t8, n8), macro(t9, n9), macro(t10, n10), macro(t11, n11), macro(t12, n12), macro(t13, n13), macro(t14, n14), macro(t15, n15), macro(t16, n16), macro(t17, n17), macro(t18, n18), macro(t19, n19)

// Implement FOR_EACH_PAIR_SEMICOLON_N for different argument counts (0 to 16 parameters = 0 to 32 arguments)
#define FOR_EACH_PAIR_SEMICOLON_0(macro, ...)

#define FOR_EACH_PAIR_SEMICOLON_2(macro, t0, n0) \
    macro(t0, n0);
#define FOR_EACH_PAIR_SEMICOLON_4(macro, t0, n0, t1, n1) \
    macro(t0, n0); macro(t1, n1);
#define FOR_EACH_PAIR_SEMICOLON_6(macro, t0, n0, t1, n1, t2, n2) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2);
#define FOR_EACH_PAIR_SEMICOLON_8(macro, t0, n0, t1, n1, t2, n2, t3, n3) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3);
#define FOR_EACH_PAIR_SEMICOLON_10(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4);
#define FOR_EACH_PAIR_SEMICOLON_12(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5);
#define FOR_EACH_PAIR_SEMICOLON_14(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6);
#define FOR_EACH_PAIR_SEMICOLON_16(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7);
#define FOR_EACH_PAIR_SEMICOLON_18(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8);
#define FOR_EACH_PAIR_SEMICOLON_20(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9);
#define FOR_EACH_PAIR_SEMICOLON_22(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10);
#define FOR_EACH_PAIR_SEMICOLON_24(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11);
#define FOR_EACH_PAIR_SEMICOLON_26(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12);
#define FOR_EACH_PAIR_SEMICOLON_28(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12); macro(t13, n13);
#define FOR_EACH_PAIR_SEMICOLON_30(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12); macro(t13, n13); macro(t14, n14);
#define FOR_EACH_PAIR_SEMICOLON_32(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12); macro(t13, n13); macro(t14, n14); macro(t15, n15);
#define FOR_EACH_PAIR_SEMICOLON_34(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12); macro(t13, n13); macro(t14, n14); macro(t15, n15); macro(t16, n16);
#define FOR_EACH_PAIR_SEMICOLON_36(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16, t17, n17) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12); macro(t13, n13); macro(t14, n14); macro(t15, n15); macro(t16, n16); macro(t17, n17);
#define FOR_EACH_PAIR_SEMICOLON_38(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16, t17, n17, t18, n18) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12); macro(t13, n13); macro(t14, n14); macro(t15, n15); macro(t16, n16); macro(t17, n17); macro(t18, n18);
#define FOR_EACH_PAIR_SEMICOLON_40(macro, t0, n0, t1, n1, t2, n2, t3, n3, t4, n4, t5, n5, t6, n6, t7, n7, t8, n8, t9, n9, t10, n10, t11, n11, t12, n12, t13, n13, t14, n14, t15, n15, t16, n16, t17, n17, t18, n18, t19, n19) \
    macro(t0, n0); macro(t1, n1); macro(t2, n2); macro(t3, n3); macro(t4, n4); macro(t5, n5); macro(t6, n6); macro(t7, n7); macro(t8, n8); macro(t9, n9); macro(t10, n10); macro(t11, n11); macro(t12, n12); macro(t13, n13); macro(t14, n14); macro(t15, n15); macro(t16, n16); macro(t17, n17); macro(t18, n18); macro(t19, n19);
