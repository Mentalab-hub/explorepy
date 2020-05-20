import exploresdk

dummy = exploresdk.ExploreSDK_Create()

list = dummy.PerformDeviceSearch()
for p in list:
    print(p.name)

bt_serial_port_manager = exploresdk.BTSerialPortBinding_Create("00:13:43:80:14:3A", 5)

bt_serial_port_manager.Connect()


bt_serial_port_manager.Send()



