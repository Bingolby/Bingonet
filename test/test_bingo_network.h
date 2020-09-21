#pragma once
#include "picotest/picotest.h"
#include "bingonet/bingonet.h"


namespace bingonet{


TEST(bingo_fc_layer, bprop) {
    bingo_network<bingo_mse, bingo_gradient_descent> nn;

    nn << bingo_fc_layer<tan_h>(4, 6)
       << bingo_fc_layer<tan_h>(6, 3);

    vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

    a[0] = 3.0; a[1] = 1.0; a[2] = -1.0; a[3] = 4.0;
    t[0] = 0.3; t[1] = 0.7; t[2] = 0.3;

    a2[0] = 1.0; a2[1] = 0.0; a2[2] = 4.0; a2[3] = 2.0;
    t2[0] = 0.6; t2[1] = 0.0; t2[2] = 0.1;

    std::vector<vec_t> data, train;

    for (int i = 0; i < 100; i++) {
        data.push_back(a);
        data.push_back(a2);
        train.push_back(t);
        train.push_back(t2);
    }
    nn.optimizer().alpha = 0.01;
    nn.train(data, train, 1, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-4);
    EXPECT_NEAR(predicted[1], t[1], 1E-4);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-4);
    EXPECT_NEAR(predicted[1], t2[1], 1E-4);

}

TEST(bingo_conv_layer, fprop) {
    typedef bingo_network<bingo_cross_entropy, bingo_gradient_descent> CNN;
    CNN nn;

    bingo_conv_layer<sigmoid> l(1, 2, 3, 5, 5);

    vec_t in(25);

    ASSERT_EQ(l.weight().size(), 18);

    std::fill(l.bias().begin(), l.bias().end(), 0.0);
    std::fill(l.weight().begin(), l.weight().end(), 0.0);

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    {
        const vec_t& out = l.forward_propagation(in, 0);

        for (auto o: out)
            EXPECT_DOUBLE_EQ(o, (bingonet::float_t)0.5);

    }

    l.weight()[0] = 0.3;  l.weight()[1] = 0.1; l.weight()[2] = 0.2;
    l.weight()[3] = 0.0;  l.weight()[4] =-0.1; l.weight()[5] =-0.1;
    l.weight()[6] = 0.05; l.weight()[7] =-0.2; l.weight()[8] = 0.05;

    l.weight()[9]  = 0.0; l.weight()[10] =-0.1; l.weight()[11] = 0.1;
    l.weight()[12] = 0.1; l.weight()[13] =-0.2; l.weight()[14] = 0.3;
    l.weight()[15] = 0.2; l.weight()[16] =-0.3; l.weight()[17] = 0.2;

    in[0] = 3;  in[1] = 2;  in[2] = 1;  in[3] = 5; in[4] = 2;
    in[5] = 3;  in[6] = 0;  in[7] = 2;  in[8] = 0; in[9] = 1;
    in[10] = 0; in[11] = 6; in[12] = 1; in[13] = 1; in[14] = 10;
    in[15] = 3; in[16] =-1; in[17] = 2; in[18] = 9; in[19] = 0;
    in[20] = 1; in[21] = 2; in[22] = 1; in[23] = 5; in[24] = 5;

    {
        const vec_t& out = l.forward_propagation(in, 0);

        EXPECT_NEAR(0.4875026, out[0], 1E-5);
        EXPECT_NEAR(0.8388910, out[1], 1E-5);
        EXPECT_NEAR(0.8099984, out[2], 1E-5);
        EXPECT_NEAR(0.7407749, out[3], 1E-5);
        EXPECT_NEAR(0.5000000, out[4], 1E-5);
        EXPECT_NEAR(0.1192029, out[5], 1E-5);
        EXPECT_NEAR(0.5986877, out[6], 1E-5);
        EXPECT_NEAR(0.7595109, out[7], 1E-5);
        EXPECT_NEAR(0.6899745, out[8], 1E-5);
    }

}

TEST(bingo_conv_layer, bprop) {
    bingo_network<bingo_cross_entropy, bingo_gradient_descent> nn;

    nn << bingo_conv_layer<sigmoid>(1, 1, 3, 5, 5);

    vec_t a(25, 0.0), t(9, 0.0);
    std::vector<vec_t> data, train;

    for (int y = 0; y < 5; y++) {
        a[5*y+3] = 1.0;
    }

    for (int y = 0; y < 3; y++) {
        t[3*y+0] = 0.0;
        t[3*y+1] = 0.5;
        t[3*y+2] = 1.0;
    }

    for (int i = 0; i < 100; i++) {
        data.push_back(a);
        train.push_back(t);
    }

    nn.train(data, train);

    vec_t predicted = nn.predict(a);
    // for(auto x : predicted)
    //     std::cout<<x<<' ';
}

}
