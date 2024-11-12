import socket
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtc
import sys
import json

class ServerApp(qtw.QWidget):
    update_client_table_signal = qtc.pyqtSignal()
    update_results_table_signal = qtc.pyqtSignal(float)

    def __init__(self):
        super().__init__()

        self.layout = qtw.QVBoxLayout()

        self.ip_label = qtw.QLabel("IP: Not set")
        self.port_label = qtw.QLabel("Port: Not set")
        self.layout.addWidget(self.ip_label)
        self.layout.addWidget(self.port_label)

        self.variant_label = qtw.QLabel("Select variant:")
        self.layout.addWidget(self.variant_label)
        self.variant_selector = qtw.QComboBox()
        self.variant_selector.addItems(["Variant 1 (sin(x)/2 Y>0)", "Variant 2 (cos(x)/2 Y<0)"])
        self.layout.addWidget(self.variant_selector)

        self.range_label = qtw.QLabel("Enter range (start, end, step):")
        self.layout.addWidget(self.range_label)
        self.range_start = qtw.QLineEdit(self)
        self.range_start.setPlaceholderText("Start")
        self.layout.addWidget(self.range_start)
        self.range_end = qtw.QLineEdit(self)
        self.range_end.setPlaceholderText("End")
        self.layout.addWidget(self.range_end)
        self.range_step = qtw.QLineEdit(self)
        self.range_step.setPlaceholderText("Step")
        self.layout.addWidget(self.range_step)

        self.assign_button = qtw.QPushButton('Assign Task', clicked=self.assign_task)
        self.layout.addWidget(self.assign_button)

        self.client_table = qtw.QTableWidget()
        self.client_table.setColumnCount(3)
        self.client_table.setHorizontalHeaderLabels(["Client IP", "Client Name", "Status"])
        self.layout.addWidget(self.client_table)

        self.disconnect_button = qtw.QPushButton('Disconnect Client', clicked=self.disconnect_client)
        self.layout.addWidget(self.disconnect_button)

        self.results_table = qtw.QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Variant", "Total Result"])
        self.layout.addWidget(self.results_table)

        self.setLayout(self.layout)
        self.setWindowTitle('Server')

        self.clients = []
        self.server_socket = None
        self.results = {}

        self.update_client_table_signal.connect(self.update_client_table)
        self.update_results_table_signal.connect(self.update_results_table)

        self.start_server()

    def get_ip_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address

    def start_server(self):
        ip_address = self.get_ip_address()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((ip_address, 0))
        server_ip, server_port = self.server_socket.getsockname()
        self.ip_label.setText(f"IP: {server_ip}")
        self.port_label.setText(f"Port: {server_port}")
        self.server_socket.listen(5)
        self.accept_clients()

    def accept_clients(self):
        while True:
            client_socket, client_address = self.server_socket.accept()
            client_name = f'Client {len(self.clients) + 1}'
            self.clients.append((client_socket, client_name, client_address[0]))
            self.update_client_table_signal.emit()
            self.handle_client(client_socket, client_name)

    def update_client_table(self):
        self.client_table.setRowCount(len(self.clients))
        for i, (client_socket, client_name, client_ip) in enumerate(self.clients):
            self.client_table.setItem(i, 0, qtw.QTableWidgetItem(client_ip))
            self.client_table.setItem(i, 1, qtw.QTableWidgetItem(client_name))
            self.client_table.setItem(i, 2, qtw.QTableWidgetItem("Connected"))

    def disconnect_client(self):
        selected_row = self.client_table.currentRow()
        if selected_row >= 0 and selected_row < len(self.clients):
            client_socket, client_name, _ = self.clients.pop(selected_row)
            client_socket.close()
            self.update_client_table_signal.emit()

    def assign_task(self):
        try:
            start = float(self.range_start.text())
            end = float(self.range_end.text())
            step = float(self.range_step.text())
            variant = self.variant_selector.currentIndex() + 1

            total_clients = len(self.clients)
            range_length = end - start
            segment_length = range_length / total_clients

            for i, (client_socket, client_name, _) in enumerate(self.clients):
                segment_start = start + i * segment_length
                segment_end = start + (i + 1) * segment_length if i < total_clients - 1 else end

                task_data = {
                    "start": segment_start,
                    "end": segment_end,
                    "step": step,
                    "option": f"Variant {variant}"
                }

                client_socket.sendall(json.dumps(task_data).encode())
        except ValueError:
            qtw.QMessageBox.critical(self, 'Input Error', 'Please enter valid numbers for range and step.')

    def handle_client(self, client_socket, client_name):
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break

                try:
                    message = json.loads(data.decode())
                    if "result" in message:
                        result = message["result"]
                        self.update_results_table_signal.emit(result)
                    else:
                        print("Received unexpected message format:", message)
                except json.JSONDecodeError:
                    try:
                        result = float(data.decode())
                        self.update_results_table_signal.emit(result)
                    except ValueError:
                        print("Received invalid result from client:", data.decode())

            except ConnectionResetError:
                break

    def update_results_table(self, result):
        variant = self.variant_selector.currentText()
        if variant not in self.results:
            self.results[variant] = 0
        self.results[variant] += result

        self.results_table.setRowCount(0)
        for variant, total_result in self.results.items():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            self.results_table.setItem(row_position, 0, qtw.QTableWidgetItem(variant))
            self.results_table.setItem(row_position, 1, qtw.QTableWidgetItem(str(total_result)))

def run_server():
    app = qtw.QApplication(sys.argv)
    main_window = ServerApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_server()