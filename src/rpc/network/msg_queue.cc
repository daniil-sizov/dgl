/*!
 *  Copyright (c) 2019 by Contributors
 * \file msg_queue.cc
 * \brief Message queue for DGL distributed training.
 */
#include <dmlc/logging.h>
#include <cstring>
#include <zlib.h>
#include "msg_queue.h"

namespace dgl {
namespace network {

using std::string;

int gzip(const void *input, size_t input_len, void *out, size_t out_size, size_t &compress_len)
{
  z_stream defstream;
  defstream.zalloc = Z_NULL;
  defstream.zfree = Z_NULL;
  defstream.opaque = Z_NULL;
                                            // setup "a" as the input and "b" as the compressed output
  defstream.avail_in = (uInt)input_len; // size of input, string + terminator
  defstream.next_in = (Bytef *)input;       // input char array
  defstream.avail_out = (uInt)out_size;     // size of output
  defstream.next_out = (Bytef *)out;        // output char array
  if (deflateInit(&defstream, Z_BEST_SPEED /*Z_BEST_COMPRESSION */) != Z_OK)
  {
    std::cout << "Error deflateInit" << std::endl;
    return 1;

  }
  auto ret = deflate(&defstream, Z_FINISH);
  if (ret < 0)
  {
    std::cout << "Error deflate " << ret << std::endl;
    return 2;

  }
  if (deflateEnd(&defstream) < 0)
  {
    std::cout << "Error deflateEnd" << std::endl;
    return 3;

  }
  compress_len = defstream.total_out;
  //std::cout << defstream.total_out << std::endl;
  return 0;

}

int ungzip(const void *input, size_t input_len, void *out, size_t out_size, size_t &total_size)
{
  z_stream infstream;
  infstream.zalloc = Z_NULL;
  infstream.zfree = Z_NULL;
  infstream.opaque = Z_NULL;
                                            // setup "b" as the input and "c" as the compressed output
  infstream.avail_in = (uInt)input_len; // size of input
  infstream.next_in = (Bytef *)input;       // input char array
  infstream.avail_out = (uInt)out_size;     // size of output
  infstream.next_out = (Bytef *)out;        // output char array
                                          // the actual DE-compression work.
  inflateInit(&infstream);
  inflate(&infstream, Z_NO_FLUSH);
  inflateEnd(&infstream);
  total_size = infstream.total_out;
  return 0;

}

int Message::unzip()
{
  if (size & GZIPED)
  {
    size &= ~GZIPED;
    int64_t len_extracted = *reinterpret_cast<int64_t *>(data);
    char *gzip_stream = data + sizeof(int64_t);
    int64_t len_gzip_stream = size - sizeof(int64_t);
    char *extr_buf = new char[len_extracted];
    size_t check_it = 0;
    if (ungzip(gzip_stream, len_gzip_stream, extr_buf, len_extracted, check_it))
    {
      log_me( "==== ERROR ungzip ====");
      show_me("==== ERROR ungzip ====");
    }

    if(this->deallocator == nullptr)
    {
       show_me(  "ERROR NULLDEALLOCATOR unzip" );
    }

    if (this->deallocator != nullptr)
      this->deallocator(this);
    //delete[] data;
    this->deallocator = DefaultMessageDeleter;
    data = extr_buf;
    log_me("Unzip from=" << size << " to=" << len_extracted << std::hex << (void *)data << std::dec);
   // size = len_extracted;
    size = check_it;
    show_me( "unzip=" << len_extracted );
  }
  return 0;

}
template <typename T, typename... U>
size_t getAddress(std::function<T(U...)> f)
{
  typedef T(fnType)(U...);
  fnType **fnPointer = f.template target<fnType *>();
  return (size_t)*fnPointer;
}

int Message::zip()
{
  //return 0;
  if (size > 3000000)
  {
    auto org_size = size;
    char *pkg_buff = new char[size + sizeof(int64_t) + 128];
   // memset(pkg_buff,0,sizeof(int64_t));
    char *gzip_stream = pkg_buff + sizeof(int64_t);
    size_t len_gziped = 0;
    if (gzip(data, size, gzip_stream, size + 128, len_gziped))
    {
      log_me("==== ERROR zip ====");
      std::cout << "==== ERROR zip ====" << std::endl;
    }
    // *reinterpret_cast<int64_t *>(pkg_buff) = len_gziped;
     *reinterpret_cast<int64_t *>(pkg_buff) = size;

    // std::function<void(dgl::network::Message *)> tt = DefaultMessageDeleter;
    // if(getAddress(this->deallocator) != getAddress(tt))
    // {
    //    std::cout << "========= FAKE Deleter ====" << std::endl;
   // }
    if (this->deallocator == nullptr)
    {
      show_me( "ERROR NULLDEALLOCATOR zip" );
    }
    if(this->deallocator != nullptr)
    this->deallocator(this);
    // delete[] data;
    data = pkg_buff;
    this->deallocator = DefaultMessageDeleter;
    log_me( "Zip from=" << size << " to=" << len_gziped << " data=" << std::hex << (void*)data << std::dec );
    size = len_gziped + sizeof(int64_t);
    show_me("zip=" << size ) ;
    size |= GZIPED;
    is_ziped=1;
    std::cout <<"[" << getMachine() <<  "] zip from=" << org_size << " to="<<(size & ~GZIPED) << " gain=" << (int)((1 - (double)(size & ~GZIPED)/(double)org_size)*100) << "%" << std::endl;
  }
  return 0;

}

MessageQueue::MessageQueue(int64_t queue_size, int num_producers) {
  CHECK_GE(queue_size, 0);
  CHECK_GE(num_producers, 0);
  queue_size_ = queue_size;
  free_size_ = queue_size;
  num_producers_ = num_producers;
}

STATUS MessageQueue::Add(Message msg, bool is_blocking) {
  // check if message is too long to fit into the queue
  if (msg.size > queue_size_) {
    LOG(WARNING) << "Message is larger than the queue.";
    return MSG_GT_SIZE;
  }
  if (msg.size <= 0) {
    LOG(WARNING) << "Message size (" << msg.size << ") is negative or zero.";
    return MSG_LE_ZERO;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (finished_producers_.size() >= num_producers_) {
    return QUEUE_CLOSE;
  }
  if (msg.size > free_size_ && !is_blocking) {
    return QUEUE_FULL;
  }
  cond_not_full_.wait(lock, [&]() {
    return msg.size <= free_size_;
  });
  // Add data pointer to queue
  queue_.push(msg);
  free_size_ -= msg.size;
  // not empty signal
  cond_not_empty_.notify_one();

  return ADD_SUCCESS;
}

STATUS MessageQueue::Remove(Message* msg, bool is_blocking) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    if (!is_blocking) {
      return QUEUE_EMPTY;
    }
    if (finished_producers_.size() >= num_producers_) {
      return QUEUE_CLOSE;
    }
  }

  cond_not_empty_.wait(lock, [this] {
    return !queue_.empty() || exit_flag_.load();
  });
  if (finished_producers_.size() >= num_producers_ && queue_.empty()) {
    return QUEUE_CLOSE;
  }

  Message old_msg = queue_.front();
  queue_.pop();
  msg->data = old_msg.data;
  msg->size = old_msg.size;
  msg->deallocator = old_msg.deallocator;
  free_size_ += old_msg.size;
  cond_not_full_.notify_one();

  return REMOVE_SUCCESS;
}

void MessageQueue::SignalFinished(int producer_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  finished_producers_.insert(producer_id);
  // if all producers have finished, consumers should be
  // waken up to get this signal
  if (finished_producers_.size() >= num_producers_) {
    exit_flag_.store(true);
    cond_not_empty_.notify_all();
  }
}

bool MessageQueue::Empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size() == 0;
}

bool MessageQueue::EmptyAndNoMoreAdd() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size() == 0 &&
         finished_producers_.size() >= num_producers_;
}

}  // namespace network
}  // namespace dgl
