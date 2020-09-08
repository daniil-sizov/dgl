/*!
 *  Copyright (c) 2019 by Contributors
 * \file tcp_socket.cc
 * \brief TCP socket for DGL distributed training.
 */
#include "tcp_socket.h"

#include <dmlc/logging.h>

#ifndef _WIN32
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif  // !_WIN32
#include <string.h>
#include <errno.h>

std::recursive_mutex &getMX()
{
  static std::recursive_mutex mx;
  return mx;
}


const char* getMachine() {
    static std::string str;
    if(!str.size())
    {
      std::lock_guard<decltype(getMX())> lock(getMX());
      if(!str.size())
      {
      char buff[32];
      gethostname(buff, sizeof(buff));
      str = buff;
      }
    }
    return str.c_str();
}
const char *getLogFileName()
{
  static std::string l;
  std::stringstream ss;
  ss << "/home/pablo/new_work/steps/log_";
  ss << getpid();
  ss << ".txt";
  l = ss.str();
  show_me( "File opened " << l );
  return l.c_str();
}

const char *getStatsFileName()
{
  static std::string l;
  std::stringstream ss;
  ss << "/home/pablo/new_work/steps/stats_";
  ss << getpid();
  ss << ".txt";
  l = ss.str();
  show_me("File opened " << l);
  return l.c_str();
}

std::ofstream &getLogStream()
{
  static std::unique_ptr<std::ofstream> file(new std::ofstream(getLogFileName()));
  return *file;
};

std::ofstream &getStatStream()
{
  static std::unique_ptr<std::ofstream> file(new std::ofstream(getStatsFileName()));
  return *file;
};

namespace dgl {
namespace network {

typedef struct sockaddr_in SAI;
typedef struct sockaddr SA;

TCPSocket::TCPSocket() {
  // init socket
  socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (socket_ < 0) {
    LOG(FATAL) << "Can't create new socket. Errno=" << errno;
  }
  log_me_this("TCPSocket::TCPSocket()");
  rcv_total = 0;
  send_total = 0;
}

TCPSocket::~TCPSocket() {
    log_me_this("TCPSocket::~TCPSocket()");

    Close();
}

bool TCPSocket::Connect(const char * ip, int port) {
  SAI sa_server;
  sa_server.sin_family      = AF_INET;
  sa_server.sin_port        = htons(port);

  int retval = 0;
  do {  // retry if EINTR failure appears
    if (0 < inet_pton(AF_INET, ip, &sa_server.sin_addr) &&
        0 <= (retval = connect(socket_, reinterpret_cast<SA*>(&sa_server),
                    sizeof(sa_server)))) {
      ip_connected = ip;
      port_connected = port;
      log_me_this("TCPSocket::Connect(ip=" << ip << "," << port << ")");
      return true;
    }
  } while (retval == -1 && errno == EINTR);

  return false;
}

bool TCPSocket::Bind(const char * ip, int port) {
  SAI sa_server;
  sa_server.sin_family      = AF_INET;
  sa_server.sin_port        = htons(port);
  int retval = 0;
  do {  // retry if EINTR failure appears
    if (0 < inet_pton(AF_INET, ip, &sa_server.sin_addr) &&
        0 <= (retval = bind(socket_, reinterpret_cast<SA*>(&sa_server),
                  sizeof(sa_server)))) {
      ip_connected = "frombind=>";
      ip_connected += ip;
      port_connected = port;
      log_me_this("TCPSocket::Bind() " << ip << ":" << port) return true;
    }
  } while (retval == -1 && errno == EINTR);

  LOG(ERROR) << "Failed bind on " << ip << ":" << port << " ,errno=" << errno;
  return false;
}

bool TCPSocket::Listen(int max_connection) {
  int retval;
  do {  // retry if EINTR failure appears
    if (0 <= (retval = listen(socket_, max_connection))) {
        log_me_this("TCPSocket::Listen(max_connection=" << max_connection)
        return true;
    }
  } while (retval == -1 && errno == EINTR);

  LOG(ERROR) << "Failed listen on socket fd: " << socket_ << " ,errno=" << errno;
  return false;
}

bool TCPSocket::Accept(TCPSocket * socket, std::string * ip, int * port) {
  int sock_client;
  SAI sa_client;
  socklen_t len = sizeof(sa_client);

  do {  // retry if EINTR failure appears
    sock_client = accept(socket_, reinterpret_cast<SA*>(&sa_client), &len);
  } while (sock_client == -1 && errno == EINTR);

  if (sock_client < 0) {
    LOG(ERROR) << "Failed accept connection on " << *ip << ":" << *port
               << " ,errno=" << errno << (errno == EAGAIN ? " SO_RCVTIMEO timeout reached" : "");
    return false;
  }

  char tmp[INET_ADDRSTRLEN];
  const char * ip_client = inet_ntop(AF_INET,
                                     &sa_client.sin_addr,
                                     tmp,
                                     sizeof(tmp));
  CHECK(ip_client != nullptr);
  ip->assign(ip_client);
  *port = ntohs(sa_client.sin_port);
  socket->socket_ = sock_client;
  socket->port_connected=*port;
  socket->ip_connected=*ip;
  log_me_this("TCPSocket::Accept(TCPSocket=" << socket << " from=" << (*ip) << ":" << (*port))
      return true;
}

#ifdef _WIN32
bool TCPSocket::SetBlocking(bool flag) {
  int result;
  u_long argp = flag ? 1 : 0;

  // XXX Non-blocking Windows Sockets apparently has tons of issues:
  // http://www.sockets.com/winsock.htm#Overview_BlockingNonBlocking
  // Since SetBlocking() is not used at all, I'm leaving a default
  // implementation here.  But be warned that this is not fully tested.
  if ((result = ioctlsocket(socket_, FIONBIO, &argp)) != NO_ERROR) {
    LOG(ERROR) << "Failed to set socket status.";
    return false;
  }
  return true;
}
#else   // !_WIN32
bool TCPSocket::SetBlocking(bool flag) {
  int opts;

  if ((opts = fcntl(socket_, F_GETFL)) < 0) {
    LOG(ERROR) << "Failed to get socket status.";
    return false;
  }

  if (flag) {
    opts |= O_NONBLOCK;
  } else {
    opts &= ~O_NONBLOCK;
  }

  if (fcntl(socket_, F_SETFL, opts) < 0) {
    LOG(ERROR) << "Failed to set socket status.";
    return false;
  }
  log_me_this("TCPSocket::SetBlocking(flag="<< flag << ")");
  return true;
}
#endif  // _WIN32

void TCPSocket::SetTimeout(int timeout) {
  #ifdef _WIN32
    timeout = timeout * 1000;  // WIN API accepts millsec
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<char*>(&timeout), sizeof(timeout));
  #else  // !_WIN32
    struct timeval tv;
    tv.tv_sec = timeout;
    tv.tv_usec = 0;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO,
               &tv, sizeof(tv));
  #endif  // _WIN32
   log_me_this("TCPSocket::SetTimeout(timeout="<< timeout << ")");
}

bool TCPSocket::ShutDown(int ways) {
  return 0 == shutdown(socket_, ways);
}

void TCPSocket::Close() {
   {
     std::lock_guard<decltype(getMX())> lock(getMX());
     stat_me("TCPSocket::Close() STATBYTES " << this->getIP() << ":" << this->getPort() << " send=" << send_total << " rcv=" << rcv_total);
     auto cnt = 0;
     for(auto& pair : m)
     {
       stat_me("MSGDETAIL " << (cnt++) << " size=" << pair.first << " count=" << pair.second  );
     }
     stat_me("-----");
   }
  if (socket_ >= 0) {
#ifdef _WIN32
    CHECK_EQ(0, closesocket(socket_));
#else   // !_WIN32
    CHECK_EQ(0, close(socket_));
#endif  // _WIN32
    socket_ = -1;
  }
  log_me_this("TCPSocket::ShutDown()");
}

int64_t TCPSocket::Send(const char * data, int64_t len_data) {
  int64_t number_send;

  do {  // retry if EINTR failure appears
    number_send = send(socket_, data, len_data, 0);
  } while (number_send == -1 && errno == EINTR);
  if (number_send == -1) {
    LOG(ERROR) << "send error: " << strerror(errno);
  }
  log_me_this("TCPSocket::Send(data=?, len_data="<< len_data << ")");
  return number_send;
}

int64_t TCPSocket::Receive(char * buffer, int64_t size_buffer) {
  int64_t number_recv;

  do {  // retry if EINTR failure appears
    number_recv = recv(socket_, buffer, size_buffer, 0);
  } while (number_recv == -1 && errno == EINTR);
  if (number_recv == -1) {
    LOG(ERROR) << "recv error: " << strerror(errno);
  }
  log_me_this("TCPSocket::Receive(buff=out?, size_buffer="<< size_buffer << ")");
  return number_recv;
}

int TCPSocket::Socket() const {
  return socket_;
}

void TCPSocket::send_complete(int64_t size)
{
       send_total+=size;
       addstat(size);
}
void TCPSocket::rcv_complete(int64_t size)
{
       rcv_total+=size;
       addstat(size);
}

void TCPSocket::addstat(int64_t size)
{
     //  catch_gdb("addstat");
      auto it = m.find(size);
      if(it==m.end())
      {
         m.insert(std::pair<int64_t,int64_t>(size,1));
      }
      else
      {
         it->second++;
      }
}

}  // namespace network
}  // namespace dgl
