function res = inSight(point, h, w,offset)

xres=(point(1)+offset)>=0 && point(1)<= (w+offset);
yres=(point(2)+offset)>=0 && point(2)<=(h+offset);
res=xres && yres;

end