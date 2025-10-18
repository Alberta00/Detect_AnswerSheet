def back_to_main(self):
        path = os.path.join(os.path.dirname(__file__), "main.py")
        if os.path.exists(path):
            QProcess.startDetached(sys.executable, [path])
            self.close()