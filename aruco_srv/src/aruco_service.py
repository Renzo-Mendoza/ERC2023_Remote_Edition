#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2022 Krzysztof Stezala
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rospy
from aruco_msgs.msg import MarkerArray
from aruco_srv.srv import *

markers50 = MarkerArray()
markers40 = MarkerArray()

def callback_service(req):
    global markers40, markers50
    resp = ArucoRequestResponse()
    if req.size==50:
        resp.markers = markers50 
    elif req.size==40:
        resp.markers = markers40
    return resp

def callback_id50(data):
    global markers50
    markers50 = data.markers

def callback_id40(data):
    global markers40
    markers40 = data.markers

def listener():
    rospy.init_node('aruco_service', anonymous=True)

    rospy.Service('aruco_erc_detect', ArucoRequest, callback_service)
    rospy.Subscriber("/aruco_marker_publisher50/markers", MarkerArray, callback_id50)
    rospy.Subscriber("/aruco_marker_publisher40/markers", MarkerArray, callback_id40)
    rospy.spin()

if __name__ == '__main__':
    listener()
