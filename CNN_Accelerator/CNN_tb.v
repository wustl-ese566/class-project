`timescale 1ns / 1ps

module ClassProject_tb();
    reg  clk;
    reg  rst;


    reg [15:0] ram_image[1024 * 1024 - 1: 0];
    reg [15:0] ram_weight[1024 * 1024 * 32 - 1: 0];

    // This part is used for debug to watch ram content in the waveform
    wire [15:0] test000 = ram_weight[0];
    wire [15:0] test001 = ram_weight[1];
    wire [15:0] test002 = ram_weight[2];
    wire [15:0] test003 = ram_weight[3];
    wire [15:0] test004 = ram_weight[4];
    wire [15:0] test005 = ram_weight[5];
    wire [15:0] test006 = ram_weight[6];
    wire [15:0] test007 = ram_weight[7];
    wire [15:0] test008 = ram_image[16];
    wire [15:0] test009 = ram_image[17];
    wire [15:0] test010 = ram_image[18];
    wire [15:0] test011 = ram_image[19];
    wire [15:0] test012 = ram_image[20];
    wire [15:0] test013 = ram_image[21];
    wire [15:0] test014 = ram_image[22];
    wire [15:0] test015 = ram_image[23];

    // Initialize memory content from "ram.bin"
    integer fd, i;
    reg [15:0] data;

    initial
    begin
        $dumpfile("ClassProject.vcd");
        $dumpvars(0, ClassProject_tb);

        fd = $fopen("weight.dat","rb");
        for (i = 0; (i < (1024 * 1024 * 8 / 4)) && ($fread(data, fd) != -1); i = i + 1)
            ram_weight[i] = {data[7:0], data[15:8]};

        $fclose(fd);

        fd = $fopen("1.dat","rb");
        for (i = 0; (i < (1024 * 1024 / 4)) && ($fread(data, fd) != -1); i = i + 1)
            ram_image[i] = {data[7:0], data[15:8]};

        $fclose(fd);

        rst = 1;
        clk = 0;

        #50
        rst = 0;

        #1000
        $finish;
    end

    always 
        #10 clk = !clk;

endmodule