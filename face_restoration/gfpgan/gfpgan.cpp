// gfpgan implemented with ncnn library

#include "gfpgan.h"

GFPGAN::GFPGAN()
{
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;
}

GFPGAN::~GFPGAN()
{
    style_conv_weights.clear();
    to_rgbs_conv_weights.clear();
    net.clear();
}

static ncnn::Mat generate_noise(const int &c, const int &h, const int &w, const float *weight)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, 1);

    ncnn::Mat noise = ncnn::Mat(w, h, c);
    for (size_t i = 0; i < noise.total(); i++)
        noise[i] = dis(gen) * weight[0];

    return noise;
}
static void relu(ncnn::Mat &in, const float &slope)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("ReLU");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, slope); // slope

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward_inplace(in, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void binary_add(const ncnn::Mat &a, const ncnn::Mat &b, ncnn::Mat &c)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("BinaryOp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0); // op_type

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}
static void binary_mul(const ncnn::Mat &a, const ncnn::Mat &b, ncnn::Mat &c)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("BinaryOp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2); // op_type

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}
static void innerproduct(const ncnn::Mat &in, const float *weight,
                         const float *bias, const int &inc, const int &num_output, ncnn::Mat &out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;
    opt.use_vulkan_compute = false;
    ncnn::Layer *op = ncnn::create_layer("InnerProduct");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, num_output);       // num_output
    pd.set(1, 1);                // bias_term
    pd.set(2, inc * num_output); // weight_data_size

    op->load_param(pd);

    // set weights
    ncnn::Mat weights[2];
    weights[0].create(inc * num_output); // weight_data
    weights[1].create(num_output);       // bias_data
    for (int i = 0; i < num_output; i++)
    {
        for (int j = 0; j < inc; j++)
            weights[0][i * inc + j] = weight[i * inc + j];
    }
    for (int i = 0; i < num_output; i++)
        weights[1][i] = bias[i];

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void concat(const ncnn::Mat &a, const ncnn::Mat &b, int axis, ncnn::Mat &c)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Concat");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}
static void convolution(const ncnn::Mat &in, const float *weight, int inc, int num_output,
                        int kernel_size, int padding, ncnn::Mat &out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;
    opt.use_vulkan_compute = false;
    ncnn::Layer *op = ncnn::create_layer("Convolution");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, num_output);                                   // num_output
    pd.set(1, kernel_size);                                  // kernel_w
    pd.set(5, 0);                                            // bias_term
    pd.set(6, inc * num_output * kernel_size * kernel_size); // weight_data_size
    pd.set(7, 1);                                            // group
    pd.set(4, padding);                                      // pad_left
    pd.set(14, padding);                                     // pad_top
    pd.set(15, padding);                                     // pad_right
    pd.set(16, padding);                                     // pad_bottom

    op->load_param(pd);

    // set weights
    ncnn::Mat weights[1];
    weights[0].create(inc * num_output * kernel_size * kernel_size); // weight_data

    for (int i = 0; i < inc * num_output * kernel_size * kernel_size; i++)
    {
        weights[0][i] = weight[i];
    }

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void scale(const ncnn::Mat &in, const float &scale, int scale_data_size, ncnn::Mat &out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Scale");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, scale_data_size); // scale_data_size
    pd.set(1, 0);               // bias_term

    op->load_param(pd);

    // set weights
    ncnn::Mat scales[1];
    scales[0].create(scale_data_size); // scale_data

    for (int i = 0; i < scale_data_size; i++)
    {
        scales[0][i] = scale;
    }

    op->load_model(ncnn::ModelBinFromMatArray(scales));

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void upsample(const ncnn::Mat &in, const float &scale, ncnn::Mat &out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);     // resize_type
    pd.set(1, scale); // height_scale
    pd.set(2, scale); // width_scale
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void clip(ncnn::Mat &in, const float &min, const float &max)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Clip");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, min); // min
    pd.set(1, max); // max

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward_inplace(in, opt);

    op->destroy_pipeline(opt);

    delete op;
}

int GFPGAN::modulated_conv(ncnn::Mat &x, ncnn::Mat &style,
                           const float *self_weight, const float *weights,
                           const float *bias, int sample_mode, int demodulate,
                           int inc, int num_output, int kernel_size, int hid_dim, ncnn::Mat &out)
{
    ncnn::Mat style_out;
    innerproduct(style, weights, bias, inc, hid_dim, style_out);

    ncnn::Mat weight = ncnn::Mat(kernel_size, kernel_size, hid_dim, num_output);

    for (int i = 0; i < weight.c; i++)
    {
        ncnn::Mat channel = weight.channel(i);
        for (int j = 0; j < weight.d; j++)
        {
            ncnn::Mat d = channel.channel(j);
            for (int k = 0; k < d.h; k++)
            {
                for (int l = 0; l < d.w; l++)
                {
                    weight[i * weight.d * weight.h * weight.w + j * weight.h * weight.w + k * weight.w + l] =
                        style_out.channel(0)[j] * self_weight[i * weight.d * weight.h * weight.w + j * weight.h * weight.w + k * weight.w + l];
                }
            }
        }
    }

    if (demodulate == 1)
    {
        ncnn::Mat demod = ncnn::Mat(num_output, 1, 1, 1);
        for (int i = 0; i < weight.c; i++)
        {
            ncnn::Mat channel = weight.channel(i);
            float sum = 0;
            for (int j = 0; j < weight.d; j++)
            {
                ncnn::Mat d = channel.channel(j);
                for (int k = 0; k < d.h; k++)
                {
                    for (int l = 0; l < d.w; l++)
                    {
                        sum += std::pow(weight[i * weight.d * weight.h * weight.w + j * weight.h * weight.w + k * weight.w + l], 2);
                    }
                }
            }
            demod[i] = 1.0 / std::sqrt(sum + 0.00000001);
        }

        for (int i = 0; i < weight.c; i++)
        {
            float *weight_data = weight.channel(i);
            for (int l = 0; l < weight.d; l++)
            {
                for (int j = 0; j < weight.h; j++)
                {
                    for (int k = 0; k < weight.w; k++)
                        weight_data[l * weight.h * weight.w + j * weight.w + k] =
                            weight_data[l * weight.h * weight.w + j * weight.w + k] * demod[i];
                }
            }
        }
    }
    if (sample_mode == 1)
    {
        upsample(x, 2.0f, x);
    }
    // ncnn::Mat conv_out;
    int paddling = std::floor(kernel_size / 2);
    convolution(x, (const float *)weight.data, hid_dim, num_output, kernel_size, paddling, out);

    return 0;
}

int GFPGAN::to_rgbs(ncnn::Mat &out, ncnn::Mat &latent, ncnn::Mat &skip,
                    int inc, int hid_dim, int num_output,
                    const float *to_rgbs_modulated_conv_weight,
                    const float *to_rgbs_modulated_conv_modulation_weight,
                    const float *to_rgbs_modulated_conv_modulation_bias,
                    const float *to_rgbs_bias)
{
    ncnn::Mat style;
    modulated_conv(out, latent, (const float *)to_rgbs_modulated_conv_weight,
                   (const float *)to_rgbs_modulated_conv_modulation_weight,
                   to_rgbs_modulated_conv_modulation_bias, 0, 0, inc, num_output, 1, hid_dim, style);

    ncnn::Mat bias = ncnn::Mat(num_output, (void *)to_rgbs_bias).reshape(1, 1, num_output);
    if (skip.empty())
        binary_add(style, bias, skip);
    else
    {
        binary_add(style, bias, style);
        upsample(skip, 2.0f, skip);
        binary_add(style, skip, skip);
    }

    return 0;
}

int GFPGAN::style_convs_modulated_conv(ncnn::Mat &x, ncnn::Mat style, int sample_mode,
                                       int demodulate, ncnn::Mat &out, int inc, int hid_dim, int num_output,
                                       const float *style_convs_modulated_conv_weight,
                                       const float *style_convs_modulated_conv_modulation_weight,
                                       const float *style_convs_modulated_conv_modulation_bias,
                                       const float *style_convs_weight,
                                       const float *style_convs_bias)
{
    ncnn::Mat conv_out;
    modulated_conv(x, style, (const float *)style_convs_modulated_conv_weight,
                   (const float *)style_convs_modulated_conv_modulation_weight,
                   style_convs_modulated_conv_modulation_bias, sample_mode, demodulate, inc, num_output, 3, hid_dim, conv_out);
    scale(conv_out, 1.4142135381698608, num_output, conv_out);

    ncnn::Mat noise = generate_noise(1, conv_out.h, conv_out.w, style_convs_weight);
    binary_add(conv_out, noise, conv_out);

    ncnn::Mat bias = ncnn::Mat(num_output, (void *)style_convs_bias).reshape(1, 1, num_output);
    binary_add(conv_out, bias, out);
    relu(out, 0.2);

    return 0;
}

int GFPGAN::load_weights(const char *model_path, std::vector<StyleConvWeights> &style_conv_weights,
                         std::vector<ToRgbConvWeights> &to_rgbs_conv_weights, ncnn::Mat &const_input)
{
    std::ifstream ifs(model_path, std::ios::binary | std::ios::in);
    if (!ifs.is_open())
    {
        return -1;
    }
    for (int i = 0; i < 15; i++)
    {
        int data_size1 = style_conv_sizes[i][0];
        int data_size2 = style_conv_sizes[i][1];
        int data_size3 = style_conv_sizes[i][2];
        int data_size4 = style_conv_sizes[i][3];
        int data_size5 = style_conv_sizes[i][4];
        int data_size = data_size1 + data_size2 + data_size3 + data_size4 + data_size5;

        StyleConvWeights weights;
        weights.data_size = data_size;
        weights.inc = style_conv_channels[i][0];
        weights.hid_dim = style_conv_channels[i][1];
        weights.num_output = style_conv_channels[i][2];
        weights.style_convs_modulated_conv_weight.resize(data_size1);
        weights.style_convs_modulated_conv_modulation_weight.resize(data_size2);
        weights.style_convs_modulated_conv_modulation_bias.resize(data_size3);
        weights.style_convs_weight.resize(data_size4);
        weights.style_convs_bias.resize(data_size5);

        ifs.read((char *)weights.style_convs_modulated_conv_weight.data(), sizeof(float) * data_size1);
        ifs.read((char *)weights.style_convs_modulated_conv_modulation_weight.data(), sizeof(float) * data_size2);
        ifs.read((char *)weights.style_convs_modulated_conv_modulation_bias.data(), sizeof(float) * data_size3);
        ifs.read((char *)weights.style_convs_weight.data(), sizeof(float) * data_size4);
        ifs.read((char *)weights.style_convs_bias.data(), sizeof(float) * data_size5);

        style_conv_weights.push_back(weights);
    }

    for (int i = 0; i < 8; i++)
    {
        int data_size1 = to_rgb_sizes[i][0];
        int data_size2 = to_rgb_sizes[i][1];
        int data_size3 = to_rgb_sizes[i][2];
        int data_size4 = to_rgb_sizes[i][3];
        int data_size = data_size1 + data_size2 + data_size3 + data_size4;

        ToRgbConvWeights weights;
        weights.data_size = data_size;
        weights.inc = to_rgb_channels[i][0];
        weights.hid_dim = to_rgb_channels[i][1];
        weights.num_output = to_rgb_channels[i][2];
        weights.to_rgbs_modulated_conv_weight.resize(data_size1);
        weights.to_rgbs_modulated_conv_modulation_weight.resize(data_size2);
        weights.to_rgbs_modulated_conv_modulation_bias.resize(data_size3);
        weights.to_rgbs_bias.resize(data_size4);

        ifs.read((char *)weights.to_rgbs_modulated_conv_weight.data(), sizeof(float) * data_size1);
        ifs.read((char *)weights.to_rgbs_modulated_conv_modulation_weight.data(), sizeof(float) * data_size2);
        ifs.read((char *)weights.to_rgbs_modulated_conv_modulation_bias.data(), sizeof(float) * data_size3);
        ifs.read((char *)weights.to_rgbs_bias.data(), sizeof(float) * data_size4);

        to_rgbs_conv_weights.push_back(weights);
    }

    int const_input_size = 4 * 4 * 512;
    std::vector<float> const_input_data;
    const_input_data.resize(const_input_size);
    ifs.read((char *)const_input_data.data(), sizeof(float) * const_input_size);
    const_input = ncnn::Mat(512 * 4 * 4, (void *)const_input_data.data()).reshape(4, 4, 512).clone();

    ifs.close();

    return 0;
}

int GFPGAN::load(const std::string &param_path, const std::string &model_path, const std::string &style_path)
{
    int ret = net.load_param(param_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open param file %s failed\n", param_path.c_str());
        return -1;
    }
    ret = net.load_model(model_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open bin file %s failed\n", model_path.c_str());
        return -1;
    }

    ret = load_weights(style_path.c_str(), style_conv_weights, to_rgbs_conv_weights, const_input);
    if (ret < 0)
    {
        fprintf(stderr, "open style file %s failed!\n", style_path.c_str());
        return -1;
    }

    return 0;
}
int GFPGAN::process(const cv::Mat &img, ncnn::Mat &outimage)
{
    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 512, 512);
    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input.1", ncnn_in);

    ncnn::Mat styles;
    ex.extract("420", styles);
    std::vector<ncnn::Mat> conditions;
    ncnn::Mat conditions0, conditions1;
    ex.extract("440", conditions0);
    ex.extract("443", conditions1);
    conditions.push_back(conditions0);
    conditions.push_back(conditions1);

    ncnn::Mat conditions2, conditions3;
    ex.extract("463", conditions2);
    ex.extract("466", conditions3);
    conditions.push_back(conditions2);
    conditions.push_back(conditions3);

    ncnn::Mat conditions4, conditions5;
    ex.extract("486", conditions4);
    ex.extract("489", conditions5);
    conditions.push_back(conditions4);
    conditions.push_back(conditions5);

    ncnn::Mat conditions6, conditions7;
    ex.extract("509", conditions6);
    ex.extract("512", conditions7);
    conditions.push_back(conditions6);
    conditions.push_back(conditions7);

    ncnn::Mat conditions8, conditions9;
    ex.extract("532", conditions8);
    ex.extract("535", conditions9);
    conditions.push_back(conditions8);
    conditions.push_back(conditions9);

    ncnn::Mat conditions10, conditions11;
    ex.extract("555", conditions10);
    ex.extract("558", conditions11);
    conditions.push_back(conditions10);
    conditions.push_back(conditions11);

    ncnn::Mat conditions12, conditions13;
    ex.extract("578", conditions12);
    ex.extract("581", conditions13);
    conditions.push_back(conditions12);
    conditions.push_back(conditions13);

    // style_conv1
    ncnn::Mat latent_0 = styles.channel(0).row_range(0, 1);
    ncnn::Mat out;
    style_convs_modulated_conv(const_input, latent_0, 0, 1, out, style_conv_weights[14].inc,
                               style_conv_weights[14].hid_dim, style_conv_weights[14].num_output,
                               style_conv_weights[14].style_convs_modulated_conv_weight.data(),
                               style_conv_weights[14].style_convs_modulated_conv_modulation_weight.data(),
                               style_conv_weights[14].style_convs_modulated_conv_modulation_bias.data(),
                               style_conv_weights[14].style_convs_weight.data(),
                               style_conv_weights[14].style_convs_bias.data());

    // to_rgb1
    ncnn::Mat latent_1 = styles.channel(0).row_range(1, 1);
    ncnn::Mat skip;
    to_rgbs(out, latent_1, skip, to_rgbs_conv_weights[7].inc,
            to_rgbs_conv_weights[7].hid_dim, to_rgbs_conv_weights[7].num_output,
            to_rgbs_conv_weights[7].to_rgbs_modulated_conv_weight.data(),
            to_rgbs_conv_weights[7].to_rgbs_modulated_conv_modulation_weight.data(),
            to_rgbs_conv_weights[7].to_rgbs_modulated_conv_modulation_bias.data(),
            to_rgbs_conv_weights[7].to_rgbs_bias.data());

    int j = 0;
    for (int i = 1; i < 14;)
    {

        ncnn::Mat latent = styles.channel(0).row_range(i, 1);
        style_convs_modulated_conv(out, latent, 1, 1, out, style_conv_weights[i - 1].inc,
                                   style_conv_weights[i - 1].hid_dim, style_conv_weights[i - 1].num_output,
                                   style_conv_weights[i - 1].style_convs_modulated_conv_weight.data(),
                                   style_conv_weights[i - 1].style_convs_modulated_conv_modulation_weight.data(),
                                   style_conv_weights[i - 1].style_convs_modulated_conv_modulation_bias.data(),
                                   style_conv_weights[i - 1].style_convs_weight.data(),
                                   style_conv_weights[i - 1].style_convs_bias.data());

        ncnn::Mat out_same = out.channel_range(0, out.c / 2);
        ncnn::Mat out_sft = out.channel_range(out.c / 2, out.c / 2);
        binary_mul(out_sft, conditions[i - 1], out_sft);
        binary_add(out_sft, conditions[i], out_sft);
        concat(out_same, out_sft, 0, out);

        latent = styles.channel(0).row_range(i + 1, 1);
        style_convs_modulated_conv(out, latent, 0, 1, out, style_conv_weights[i].inc,
                                   style_conv_weights[i].hid_dim, style_conv_weights[i].num_output,
                                   style_conv_weights[i].style_convs_modulated_conv_weight.data(),
                                   style_conv_weights[i].style_convs_modulated_conv_modulation_weight.data(),
                                   style_conv_weights[i].style_convs_modulated_conv_modulation_bias.data(),
                                   style_conv_weights[i].style_convs_weight.data(),
                                   style_conv_weights[i].style_convs_bias.data());

        latent = styles.channel(0).row_range(i + 2, 1);
        to_rgbs(out, latent, skip, to_rgbs_conv_weights[j].inc,
                to_rgbs_conv_weights[j].hid_dim, to_rgbs_conv_weights[j].num_output,
                to_rgbs_conv_weights[j].to_rgbs_modulated_conv_weight.data(),
                to_rgbs_conv_weights[j].to_rgbs_modulated_conv_modulation_weight.data(),
                to_rgbs_conv_weights[j].to_rgbs_modulated_conv_modulation_bias.data(),
                to_rgbs_conv_weights[j].to_rgbs_bias.data());

        i += 2;
        j += 1;
    }
    clip(skip, -1.0f, 1.0f);
    outimage = skip;

    return 0;
}
