/*!
 *  Copyright (c) 2019 by Contributors
 * \file tcp_socket.h
 * \brief TCP socket for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_TCP_SOCKET_H_
#define DGL_RPC_NETWORK_TCP_SOCKET_H_

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "Ws2_32.li b")
#else   // !_WIN32
#include <sys/socket.h>
#endif  // _WIN32
#include <string>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <memory>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <map>
std::recursive_mutex& getMX();

#ifndef MSG_EOF
#ifdef MSG_FIN
#define MSG_EOF MSG_FIN
#endif
#endif

#ifdef MSG_EOF
// T/TCP  using
#endif
const char*  getLogFileName();
const char *getStatsFileName();
const char *getMachine();
std::ofstream&getLogStream();
std::ofstream &getStatStream();
//#define show_me(x) std::cout << x << std::endl;
#define  catch_gdb(msg) {  \
   char buff[32]; \
   gethostname(buff,sizeof(buff)); \
   std::cout << "*****CatchGDB****  ["<< buff << "]" << msg << " pid=" << getpid() <<" ppid="<< getppid() << std::endl; \
   static int local=1; \
   while(local) {}  \
  }
#ifndef show_me(x)
#define show_me(x)
#endif
#define log_me(x)                                                                                                                      \
  { \
    if(std::getenv("NODE_LABEL")) {                                                                                                  \
    std::lock_guard<decltype(getMX())> lock(getMX());                                                                                  \
    auto& stream = getLogStream();                                                                                                        \
    stream << "[deb][" << std::hex << std::this_thread::get_id() << std::dec << "|" << getpid() << "] " << x << std::dec << std::endl; } \
  }
#define log_me_this(x) log_me( std::hex << "this=" << this  << std::dec << " "<< x)

#define stat_me(x)                                                                                                                        \
  {                                                                                                                                      \
    if (1)                                                                                                       \
    {                                                                                                                                    \
      std::lock_guard<decltype(getMX())> lock(getMX());                                                                                  \
      auto &stream = getStatStream();                                                                                                     \
      stream << "[deb][" << std::hex << std::this_thread::get_id() << std::dec << "|" << getpid() << "] " << x << std::dec << std::endl; \
    }                                                                                                                                    \
  }

namespace dgl {
namespace network {

/*!
 * \brief TCPSocket is a simple wrapper around a socket.
 * It supports only TCP connections.
 */
class TCPSocket {
 public:
  /*!
   * \brief TCPSocket constructor
   */
  TCPSocket();

  /*!
   * \brief TCPSocket deconstructor
   */
  ~TCPSocket();

  /*!
   * \brief Connect to a given server address
   * \param ip ip address
   * \param port end port
   * \return true for success and false for failure
   */
  bool Connect(const char * ip, int port);

  /*!
   * \brief Bind on the given IP and PORT
   * \param ip ip address
   * \param port end port
   * \return true for success and false for failure
   */
  bool Bind(const char * ip, int port);

  /*!
   * \brief listen for remote connection
   * \param max_connection maximal connection
   * \return true for success and false for failure
   */
  bool Listen(int max_connection);

  /*!
   * \brief wait doe a new connection
   * \param socket new SOCKET will be stored to socket
   * \param ip_client new IP will be stored to ip_client
   * \param port_client new PORT will be stored to port_client
   * \return true for success and false for failure
   */
  bool Accept(TCPSocket * socket,
              std::string * ip_client,
              int * port_client);

  /*!
   * \brief SetBlocking() is needed refering to this example of epoll:
   * http://www.kernel.org/doc/man-pages/online/pages/man4/epoll.4.html
   * \param flag flag for blocking
   * \return true for success and false for failure
   */
  bool SetBlocking(bool flag);

  /*!
   * \brief Set timeout for socket
   * \param timeout seconds timeout
   */
  void SetTimeout(int timeout);

  /*!
   * \brief Shut down one or both halves of the connection.
   * \param ways ways for shutdown
   * If ways is SHUT_RD, further receives are disallowed.
   * If ways is SHUT_WR, further sends are disallowed.
   * If ways is SHUT_RDWR, further sends and receives are disallowed.
   * \return true for success and false for failure
   */
  bool ShutDown(int ways);

  /*!
   * \brief close socket.
   */
  void Close();

  /*!
   * \brief Send data.
   * \param data data for sending
   * \param len_data length of data
   * \return return number of bytes sent if OK, -1 on error
   */
  int64_t Send(const char * data, int64_t len_data);

  /*!
   * \brief Receive data.
   * \param buffer buffer for receving
   * \param size_buffer size of buffer
   * \return return number of bytes received if OK, -1 on error
   */
  int64_t Receive(char * buffer, int64_t size_buffer);

  /*!
   * \brief Get socket's file descriptor
   * \return socket's file descriptor
   */
  int Socket() const;
  const std::string& getIP() { return ip_connected; }
  int getPort() { return port_connected; }
  void send_complete(int64_t size);
  void rcv_complete(int64_t size);
  void addstat(int64_t size);
private:
  /*!
   * \brief socket's file descriptor
   */
  int socket_;
  std::string ip_connected;
  int port_connected;
  int64_t rcv_total;
  int64_t send_total;
  std::map<int64_t,int64_t> m;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_TCP_SOCKET_H_
