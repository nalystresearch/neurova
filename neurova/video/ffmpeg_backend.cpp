/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

namespace py = pybind11;

class FFmpegReader {
public:
    explicit FFmpegReader(const std::string& path)
        : path_(path) {
        open();
    }

    ~FFmpegReader() { close(); }

    bool is_open() const { return fmt_ctx_ != nullptr; }
    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }
    int64_t frame_count() const { return frame_count_; }

    py::array_t<uint8_t> read() {
        if (!is_open()) {
            return py::array_t<uint8_t>();
        }

        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        bool produced = false;

        while (!produced) {
            int ret = av_read_frame(fmt_ctx_, packet);
            if (ret < 0) {
                if (ret == AVERROR_EOF) {
                    avcodec_send_packet(codec_ctx_, nullptr);
                    while ((ret = avcodec_receive_frame(codec_ctx_, frame)) == 0) {
                        auto arr = frame_to_array(frame);
                        av_frame_free(&frame);
                        av_packet_free(&packet);
                        return arr;
                    }
                }
                av_frame_free(&frame);
                av_packet_free(&packet);
                return py::array_t<uint8_t>();
            }

            if (packet->stream_index != video_stream_index_) {
                av_packet_unref(packet);
                continue;
            }

            ret = avcodec_send_packet(codec_ctx_, packet);
            if (ret < 0) {
                av_packet_unref(packet);
                continue;
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx_, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    break;
                }
                auto arr = frame_to_array(frame);
                av_packet_unref(packet);
                av_frame_unref(frame);
                av_frame_free(&frame);
                av_packet_free(&packet);
                produced = true;
                return arr;
            }

            av_packet_unref(packet);
        }

        av_frame_free(&frame);
        av_packet_free(&packet);
        return py::array_t<uint8_t>();
    }

    bool seek(int64_t frame_index) {
        if (!is_open() || frame_index < 0) {
            return false;
        }
        int64_t timestamp = av_rescale_q(frame_index, av_make_q(1, fps_ > 0 ? static_cast<int>(fps_) : 30), stream_->time_base);
        if (av_seek_frame(fmt_ctx_, video_stream_index_, timestamp, AVSEEK_FLAG_BACKWARD) < 0) {
            return false;
        }
        avcodec_flush_buffers(codec_ctx_);
        return true;
    }

    void close() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
        }
        if (fmt_ctx_) {
            avformat_close_input(&fmt_ctx_);
        }
        stream_ = nullptr;
    }

private:
    void open() {
        if (avformat_open_input(&fmt_ctx_, path_.c_str(), nullptr, nullptr) < 0) {
            throw std::runtime_error("Failed to open input" );
        }
        if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
            throw std::runtime_error("Failed to find stream info");
        }
        video_stream_index_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_index_ < 0) {
            throw std::runtime_error("Failed to find video stream");
        }
        stream_ = fmt_ctx_->streams[video_stream_index_];
        AVCodecParameters* codecpar = stream_->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            throw std::runtime_error("Unsupported codec");
        }
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            throw std::runtime_error("Failed to alloc codec context");
        }
        if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
            throw std::runtime_error("Failed to copy codec parameters");
        }
        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            throw std::runtime_error("Failed to open codec");
        }

        width_ = codec_ctx_->width;
        height_ = codec_ctx_->height;
        fps_ = av_q2d(stream_->avg_frame_rate);
        if (fps_ <= 0.0) {
            fps_ = av_q2d(stream_->r_frame_rate);
        }
        if (fps_ <= 0.0) {
            fps_ = 30.0;
        }
        if (stream_->nb_frames > 0) {
            frame_count_ = static_cast<int64_t>(stream_->nb_frames);
        } else if (stream_->duration > 0) {
            frame_count_ = static_cast<int64_t>(stream_->duration * fps_ * av_q2d(stream_->time_base));
        } else {
            frame_count_ = 0;
        }

        sws_ctx_ = sws_getContext(
            width_,
            height_,
            codec_ctx_->pix_fmt,
            width_,
            height_,
            AV_PIX_FMT_RGB24,
            SWS_BILINEAR,
            nullptr,
            nullptr,
            nullptr);
        if (!sws_ctx_) {
            throw std::runtime_error("Failed to create sws context");
        }
        rgb_buffer_.resize(static_cast<size_t>(width_) * height_ * 3);
    }

    py::array_t<uint8_t> frame_to_array(AVFrame* frame) {
        if (!sws_ctx_) {
            return py::array_t<uint8_t>();
        }
        uint8_t* dst_data[4] = {rgb_buffer_.data(), nullptr, nullptr, nullptr};
        int dst_linesize[4] = {width_ * 3, 0, 0, 0};
        sws_scale(
            sws_ctx_,
            frame->data,
            frame->linesize,
            0,
            height_,
            dst_data,
            dst_linesize);
        py::array_t<uint8_t> array({height_, width_, 3});
        std::memcpy(array.mutable_data(), rgb_buffer_.data(), rgb_buffer_.size());
        return array;
    }

    std::string path_;
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    int video_stream_index_ = -1;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    int64_t frame_count_ = 0;
    std::vector<uint8_t> rgb_buffer_;
};

PYBIND11_MODULE(ffmpeg_backend, m) {
    m.attr("__doc__") = "FFmpeg/Libav backend for Neurova video decoding";

    py::class_<FFmpegReader>(m, "FFmpegReader")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def("is_open", &FFmpegReader::is_open)
        .def("read", &FFmpegReader::read)
        .def("seek", &FFmpegReader::seek)
        .def("close", &FFmpegReader::close)
        .def_property_readonly("width", &FFmpegReader::width)
        .def_property_readonly("height", &FFmpegReader::height)
        .def_property_readonly("fps", &FFmpegReader::fps)
        .def_property_readonly("frame_count", &FFmpegReader::frame_count);
}
