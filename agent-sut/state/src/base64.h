#pragma once
#include <string>

inline std::string base64_encode(const unsigned char* bytes, size_t len){
  static const char* tbl =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out; out.reserve(((len+2)/3)*4);
  unsigned char a3[3]; unsigned char a4[4]; size_t i=0;
  while(len--){
    a3[i++] = *bytes++;
    if(i==3){
      a4[0]=(a3[0]&0xfc)>>2;
      a4[1]=((a3[0]&0x03)<<4)+((a3[1]&0xf0)>>4);
      a4[2]=((a3[1]&0x0f)<<2)+((a3[2]&0xc0)>>6);
      a4[3]=a3[2]&0x3f;
      for(i=0;i<4;i++) out.push_back(tbl[a4[i]]);
      i=0;
    }
  }
  if(i){
    for(size_t j=i;j<3;j++) a3[j]=0;
    a4[0]=(a3[0]&0xfc)>>2;
    a4[1]=((a3[0]&0x03)<<4)+((a3[1]&0xf0)>>4);
    a4[2]=((a3[1]&0x0f)<<2)+((a3[2]&0xc0)>>6);
    a4[3]=a3[2]&0x3f;
    for(size_t j=0;j<i+1;j++) out.push_back(tbl[a4[j]]);
    while(i++<3) out.push_back('=');
  }
  return out;
}
