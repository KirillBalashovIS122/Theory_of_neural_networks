import socket
import sys
import json
import PyQt5.QtWidgets as qtw
import math
import logging

logging.basicConfig(filename='client.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClientApp(qtw.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = qtw.QVBoxLayout()

        self.name_label = qtw.QLabel("Client Name:")
        self.layout.addWidget(self.name_label)
        self.name_input = qtw.QLineEdit(self)
        self.layout.addWidget(self.name_input)

        self.ip_label = qtw.QLabel("Server IP:")
        self.layout.addWidget(self.ip_label)
        self.ip_input = qtw.QLineEdit(self)
        self.layout.addWidget(self.ip_input)

        self.port_label = qtw.QLabel("Server Port:")
        self.layout.addWidget(self.port_label)
        self.port_input = qtw.QLineEdit(self)
        self.layout.addWidget(self.port_input)

        self.connect_button = qtw.QPushButton('Connect', clicked=self.connect_to_server)
        self.layout.addWidget(self.connect_button)

        self.disconnect_button = qtw.QPushButton('Disconnect', clicked=self.disconnect_from_server)
        self.layout.addWidget(self.disconnect_button)

        self.exit_button = qtw.QPushButton('Exit', clicked=self.close)
        self.layout.addWidget(self.exit_button)

        self.status_label = qtw.QLabel("Status: Not connected")
        self.layout.addWidget(self.status_label)

        self.progress_bar = qtw.QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.result_label = qtw.QLabel("Computed Result: N/A")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)
        self.setWindowTitle('Client')

        self.client_socket = None

    def connect_to_server(self):
        server_ip = self.ip_input.text()
        server_port = int(self.port_input.text())
        client_name = self.name_input.text()

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((server_ip, server_port))
            self.status_label.setText("Status: Connected")
            logging.info(f"Connected to server {server_ip}:{server_port}")
            self.receive_data(client_name)
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            logging.error(f"Failed to connect to server {server_ip}:{server_port}: {str(e)}")

    def receive_data(self, client_name):
        while True:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    break
                task_data = json.loads(data.decode())
                result = self.perform_calculations_with_progress(task_data)
                self.result_label.setText(f"Computed Result: {result}")
                result_json = json.dumps({"result": result, "client_name": client_name})
                self.client_socket.sendall(result_json.encode())
                logging.info(f"Sent result to server: {result}")
            except json.JSONDecodeError:
                self.status_label.setText("Status: Error - Invalid JSON received.")
                logging.error("Invalid JSON received from server.")
                break
            except Exception as e:
                self.status_label.setText(f"Status: Error - {str(e)}")
                logging.error(f"Error receiving data from server: {str(e)}")
                break

    def perform_calculations_with_progress(self, data):
        start = data['start']
        end = data['end']
        step = data['step']
        option = data['option']

        result = 0
        total_steps = int((end - start) / step)
        current_step = 0

        if "Variant 1" in option:
            while start <= end:
                y = math.sin(start)
                if y > 0:
                    result += abs(y * step)
                start += step
                current_step += 1
                self.progress_bar.setValue(int((current_step / total_steps) * 100))
        elif "Variant 2" in option:
            while start <= end:
                y = math.cos(start)
                if y < 0:
                    result += abs(y * step)
                start += step
                current_step += 1
                self.progress_bar.setValue(int((current_step / total_steps) * 100))

        logging.info(f"Calculated result: {result} for option: {option}")
        return result

    def disconnect_from_server(self):
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
            self.status_label.setText("Status: Disconnected")
            logging.info("Disconnected from server")

def run_client():
    app = qtw.QApplication(sys.argv)
    client_app = ClientApp()
    client_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_client()