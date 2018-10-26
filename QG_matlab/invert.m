function [ph1,ph2]=invert(zh1,zh2,a11,a12,a21,a22)
  ph1=a11.*zh1+a12.*zh2;
  ph2=a21.*zh1+a22.*zh2;
