import tensorflow as tf
x = tf . constant ('Hello world !')
y1 = tf . constant (2 , dtype = tf . int32 )
y2 = tf . constant (1 , dtype = tf . int32 )
z = y1 + y2

sess = tf . Session ()
x_eval = sess . run ( x )
print ( x_eval )
z_eval = sess . run ( z )
print ( z_eval )
y1_eval , y2_eval = sess . run ([ y1 , y2 ])
print ( y1_eval , y2_eval )
