#pragma once
#include "picotest/picotest.h"
#include "bingonet/bingonet.h"
#include "bingonet/layers/bingo_input_layer.h"


namespace bingonet{
TEST(bingo_fc_layer, init)
{
    bingo_fc_layer<tan_h> l1(100, 100);
    bingo_fc_layer<tan_h> l2(100, 100);

    l1.init_weight();
    l2.init_weight();

    //serialization_test(l1, l2);
}

TEST(bingo_input_layer, init)
{
    bingo_input_layer();

    //serialization_test(l1, l2);
}

}
