#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    printf("Running Google Test from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}