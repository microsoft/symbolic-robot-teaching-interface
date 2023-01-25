import PySimpleGUI as sg
import cv2


class preview_gui:
    def __init__(self, fp_video, trasncript=""):
        self.fp = fp_video
        self.cap = cv2.VideoCapture(str(self.fp))
        self.transcript = trasncript

        self.ret, self.f_frame = self.cap.read()
        if self.ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_count = 0
            self.s_frame = 0
            self.e_frame = self.total_count
            self.stop_flg = False
            cv2.namedWindow("Video preview")
        else:
            sg.Popup("load error")
            return

    def run(self):
        layout = [
            [
                sg.Text("Speed", size=(6, 1), font='Helvetica 14'),
                sg.Slider(
                    (5, 100),
                    10,
                    5,
                    orientation='h',
                    key='-SLIDER-',
                    enable_events=True,
                    font='Helvetica 14'
                )
            ],
            [sg.HorizontalSeparator()],
            [
                sg.Text("Verbal input", font='Helvetica 14'),
                sg.InputText(self.transcript, key='-VERBAL-',
                             font='Helvetica 14'),
                sg.Submit("Confirm", key="Confirm", font='Helvetica 14')
            ],
        ]
        window = sg.Window('Video confirmation', layout, location=(0, 300))

        self.event, values = window.read(timeout=0)

        while True:
            self.event, values = window.read(
                timeout=values["-SLIDER-"]
            )

            if self.event != "__TIMEOUT__":
                print(self.event)

            # Exit condition
            if self.event in ('Exit', sg.WIN_CLOSED, None):
                self.transcript = ""
                break
            if self.event in ('Confirm'):
                print(values['-VERBAL-'])
                verbal_input = values['-VERBAL-']
                self.transcript = verbal_input
                break

            # Loop video
            if self.frame_count >= self.e_frame:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                self.frame_count = self.s_frame
                continue

            if ((self.stop_flg and self.event == "__TIMEOUT__")):
                continue

            self.ret, self.frame = self.cap.read()
            # resize frame for vizualization
            scale = 0.3
            self.frame = cv2.resize(
                self.frame, (int(self.width * scale),
                             int(self.height * scale)),
                interpolation=cv2.INTER_NEAREST)
            self.valid_frame = int(self.frame_count - self.s_frame)

            if not self.ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.s_frame)
                self.frame_count = self.s_frame
                continue

            cv2.putText(self.frame,
                        str("framecount: {0:.0f}".format(self.frame_count)),
                        (15,
                            20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (240,
                            230,
                            0),
                        1,
                        cv2.LINE_AA)
            cv2.putText(self.frame,
                        str("time: {0:.1f} sec".format(
                            self.frame_count / self.fps)),
                        (15,
                            40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (240,
                            230,
                            0),
                        1,
                        cv2.LINE_AA)

            cv2.imshow("Video preview", self.frame)
            cv2.moveWindow("Video preview", 0, 0)
            if self.stop_flg:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
            else:
                self.frame_count += 1

        cv2.destroyWindow("Video preview")
        self.cap.release()
        window.close()
        return self.transcript
