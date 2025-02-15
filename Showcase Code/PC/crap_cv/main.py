import ctypes
import threading
import signal
import pickle
import os

import cv2
import cv2.aruco as aruco
import numpy as np
from tkinter import Frame, Tk, Label, IntVar
from PIL import Image, ImageTk
from tkinter.ttk import Button, Checkbutton
from flask import Flask

from util import corner_point, extrapolate_points, draw_points, interpolate_points


ROOK, BISHOP, KNIGHT, KING, QUEEN, PAWN = range(6)
BLACK, WHITE = range(2)
NAME = {
    ROOK: "Rook",
    BISHOP: "Bishop",
    KNIGHT: "Knight",
    KING: "King",
    QUEEN: "Queen",
    PAWN: "Pawn",
}

SHOW_MASK = True
SHOW_ARMASK = True
SHOW_CHESS_MARKERS = True
SHOW_EXTRAPOLATED = True
SHOW_BOARD = True
SHOW_QUICK_CAL = True
SHOW_REGIONS = True

CORNERS = [
    403, 634, 666, 76
]


WINDOW_MASK = "Mask"
WINDOW_ARMASK = "ARUco Mask"
WINDOW_MARKERS = "Markers"
WINDOW_EXTRAP_BOARD = "Extrapolated Board"
WINDOW_EXTRA_BOARD = "Extracted Board"
WINDOW_QUICK_CAL = "Quick Callibration"
WINDOW_SQUARES = "Squares"

TK_LAYOUT = (
    (WINDOW_MASK, WINDOW_ARMASK),
    (WINDOW_MARKERS, WINDOW_EXTRAP_BOARD, WINDOW_EXTRA_BOARD),
    (WINDOW_QUICK_CAL, WINDOW_SQUARES),
)


POINTS_CACHE = "points.pkl"

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

PREVIEW_SIZE = (256, 256)
PREVIEW_SIZE = (300, 300)


def test_port(port, non_working_ports, working_ports, available_ports):
    camera = cv2.VideoCapture(port, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    if not camera.isOpened():
        non_working_ports.append(port)
        print(f"Port {port} is not working.")
    else:
        could_read, _ = camera.read()
        w = camera.get(3)
        h = camera.get(4)

        if could_read:
            print(f"Port {port} is working and reads images ({w} x {h})")
            working_ports.append((port, w, h))
        else:
            print(f"Port {port} for camera ({w} x {h}) is present but does not read")
            available_ports.append((port, w, h))

    camera.release()


def terminate_thread(thread):
    if not thread.is_alive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def list_ports():
    non_working_ports = []
    port = 0
    working_ports = []
    available_ports = []

    while len(non_working_ports) < 6:
        test_thread = threading.Thread(
            target=test_port,
            args=(port, non_working_ports, working_ports, available_ports),
        )
        test_thread.start()
        test_thread.join(10)
        if test_thread.is_alive():
            print(f"Port {port} timed out")
            terminate_thread(test_thread)
            non_working_ports.append(port)

        port += 1

    print("=============================")
    for port, w, h in working_ports:
        print(f"{port}: {w}x{h}")

    return available_ports, working_ports, non_working_ports


class Main:
    PIECES_WAIT = 15
    ALLOCATIONS = { 
        #allocate rows as 1st matrix, column as 2nd matrix
        (0, 0): (ROOK, WHITE),
        (1, 0): (KNIGHT, WHITE),
        (2, 0): (BISHOP, WHITE),
        (3, 0): (KING, WHITE),
        (4, 0): (QUEEN, WHITE),
        (5, 0): (BISHOP, WHITE),
        (6, 0): (KNIGHT, WHITE),
        (7, 0): (ROOK, WHITE),
        (0, 1): (PAWN, WHITE),
        (1, 1): (PAWN, WHITE),
        (2, 1): (PAWN, WHITE),
        (3, 1): (PAWN, WHITE),
        (4, 1): (PAWN, WHITE),
        (5, 1): (PAWN, WHITE),
        (6, 1): (PAWN, WHITE),
        (7, 1): (PAWN, WHITE),
        (0, 6): (PAWN, BLACK),
        (1, 6): (PAWN, BLACK),
        (2, 6): (PAWN, BLACK),
        (3, 6): (PAWN, BLACK),
        (4, 6): (PAWN, BLACK),
        (5, 6): (PAWN, BLACK),
        (6, 6): (PAWN, BLACK),
        (7, 6): (PAWN, BLACK),
        (0, 7): (ROOK, BLACK),
        (1, 7): (KNIGHT, BLACK),
        (2, 7): (BISHOP, BLACK),
        (3, 7): (KING, BLACK),
        (4, 7): (QUEEN, BLACK),
        (5, 7): (BISHOP, BLACK),
        (6, 7): (KNIGHT, BLACK),
        (7, 7): (ROOK, BLACK),
    }
    #use default camera.
    def __init__(self, camera=0):
        self.images = {}
        #aruco is the QR code library. Gets aruco markers.
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters_create()
        #start video capture.
        #set video height and width, cropping individual sqaures
        self.vid = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        self._make_tk()

        self.pieces = {}
        self.piece_allocations = {}

        #width of the square in chess square? Fed it into the position matrix
        self.square_w = 100

        self.grey = None
        self.points = None

        # Really don't care to be honest
        #get rid of error messages
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        #creates web app using Flask
        self.app = Flask(__name__)
        self.app.route("/")(self.flask_route)
        
        #running - true (intializer)
        self.running = True

    #tk is for the graphical user interface (ignore this for now)
    def _make_tk(self):
        self.tk = Tk()
        self.tk.title("CRAP Eyesight")
        self.tk.resizable(False, False)
        self.tk.protocol("WM_DELETE_WINDOW", self._tk_delete)

        control_frame = Frame(self.tk)
        control_frame.grid(row=0, column=0, sticky="N")
        self._make_control_tk(control_frame)

        images_frame = Frame(self.tk)
        images_frame.grid(row=0, column=1)
        self._make_image_tk(images_frame)

    def _tk_delete(self):
        self.running = False
        self.tk = None

    def _make_control_tk(self, frame):
        Button(frame, text="Save Calibration", command=self.save).pack()
        Button(frame, text="Load Calibration", command=self.load).pack()
        Button(frame, text="Bake Allocations", command=self.bake_allocations).pack()

        self.live_cal = IntVar()
        self.live_cal.set(1)
        f = Frame(frame)
        Label(f, text="Live Calibration").pack(side="left")
        Checkbutton(f, variable=self.live_cal, state="on").pack(side="right")
        f.pack()

        self.quick_cal = IntVar()
        self.quick_cal.set(0)
        f = Frame(frame)
        Label(f, text="Quick Calibration").pack(side="left")
        Checkbutton(f, variable=self.quick_cal, state="on").pack(side="right")
        f.pack()

        self.sharpen = IntVar()
        self.sharpen.set(1)
        f = Frame(frame)
        Label(f, text="Sharpen Image").pack(side="left")
        Checkbutton(f, variable=self.sharpen, state="on").pack(side="right")
        f.pack()

    def _make_image_tk(self, frame):
        self.img_tk_labels = {}

        for row, items in enumerate(TK_LAYOUT):
            for col, name in enumerate(items):
                lab = Label(frame)
                lab.grid(row=row * 2, column=col)
                Label(frame, text=name).grid(row=row * 2 + 1, column=col)

                self.img_tk_labels[name] = lab

    def show(self, name, image):
        image = cv2.resize(image, PREVIEW_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.images[name] = image

    def render_tk(self):
        for name, lab in self.img_tk_labels.items():
            img = self.images.get(name)
            if img is None:
                continue

            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im, master=self.tk)

            lab.configure(image=imgtk)
            lab.image = imgtk

        self.tk.update()

    #associate each piece to a position on the board. self.pieces[i] etc.
    #extract these parts. [i,j] where i is column and j is arrays.
    def bake_allocations(self):
        self.piece_allocations.clear()
        for i in self.pieces:
            allocation = self.ALLOCATIONS.get((self.pieces[i][0], self.pieces[i][1]))
            if allocation is not None:
                self.piece_allocations[i] = allocation
    #save position
    def save(self):
        with open(POINTS_CACHE, "wb") as f:
            pickle.dump(self.points, f) #save into file
    #load position
    def load(self):
        if os.path.exists(POINTS_CACHE):
            with open(POINTS_CACHE, "rb") as f:
                self.points = pickle.load(f) #opens for reading in binary mode.
    #remove this part - aruco related.
    def identify_aruco(self, img, do_mask=True):
        if do_mask:
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, masked = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if SHOW_ARMASK:
                self.show(WINDOW_ARMASK, masked)
        else:
            masked = img

        corners, ids, _ = aruco.detectMarkers(
            masked, self.aruco_dict, parameters=self.aruco_params
        )
        return corners, ids

    
    def get_frame(self):
        ret, img = self.vid.read()
        if not ret:
            print("TITSUP")
            quit()

        #gaussian blur - identify it even if the image is blur.
        #enhance image data via Gaussian bur and add Weight
        if self.sharpen.get():
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            return cv2.addWeighted(img, 1.5, blurred, -0.5, 0, blurred)

        return img
    #creates binary mark. Grayscale
    def get_mask(self, img):
        self.grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #automatically determine the threshold of the mask. In order to detect black and white.
        _, msk = cv2.threshold(
            self.grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        if SHOW_MASK:
            self.show(WINDOW_MASK, msk)

        return msk
    #create rectangular region, transform the image into wrapped image (undistorted,, front-facing concept)
    def extract(self, img, points, w=50, h=50, p=0):
        out_pts = np.float32(
            [(p, p), (w - 1 - p, p), (p, h - 1 - p), (w - 1 - p, h - 1 - p)]
        )
        #getPerspective..(src,dst,inputArray,size, int flags)
        matrix = cv2.getPerspectiveTransform(points, out_pts)
        #return the undistored image using bilinear interpolation - resampling method that uses distance weight average
        #of 4 nearest pixels to estimate new pixel values (new squares).
        return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    #relates to the chess square dimensions. Cut up board intos sqaures and stich them up together after that.
    #crop out 8x8
    def get_8x8_regions(self, img, points):
        regions = [[None for _ in range(8)] for _ in range(8)]
        #2 loop times for rows and columns. X for rows and Y for columns.
        for x in range(8):
            for y in range(8):
                regions[x][y] = self.extract(
                    img,
                    np.float32(
                        [
                            points[x][y],
                            points[x + 1][y],
                            points[x][y + 1],
                            points[x + 1][y + 1],
                        ]
                    ),
                    self.square_w,
                    self.square_w,
                )
        return regions
    #stich 8x8 squares
    def stitch_8x8(self, regions):
        full_w, full_h = self.square_w * 8 + 90, self.square_w * 8 + 90
        out = np.zeros((full_w, full_h, 3), dtype=np.uint8)
        out[:] = 255
        for x in range(8):
            for y in range(8):
                out[
                    y * (self.square_w + 10) + 10: (y + 1) * (self.square_w + 10),
                    x * (self.square_w + 10) + 10: (x + 1) * (self.square_w + 10),
                ] = regions[x][y]
        return out
    # uses QR codes to identify each square (commented out)
    def process_regions(self, regions):
        # for x in range(8):
        #     for y in range(8):
        #         self.identify_aruco(regions[x][y])
        #if show is true -> stich up the 8x8 sqaures.
        if SHOW_REGIONS:
            stitched = self.stitch_8x8(regions)
            self.show(WINDOW_SQUARES, stitched)

    def update_cal(self, img):
        msk = self.get_mask(img)
        # Extract chess-board
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        dlt = cv2.dilate(msk, krn, iterations=5)
        res = 255 - cv2.bitwise_and(dlt, msk)

        # Displaying chess-board features
        res = np.uint8(res)
        ret, corners = cv2.findChessboardCorners(
            res,
            (7, 7),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ret:
            dx = corners[0][0][0] - corners[6][0][0]
            dy = corners[0][0][1] - corners[6][0][1]
            if abs(dy) > abs(dx):
                corners = np.reshape(corners, (7, 7, 1, 2))
                corners = np.rot90(corners, 1, axes=(0, 1))
                corners = np.reshape(corners, (49, 1, 2))
            dy = corners[0][0][1] - corners[45][0][1]
            if dy > 0:
                corners = corners[::-1]

        objp = np.zeros((7 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * 30

        if ret:
            #update self.points if found the corners. self.points contain positions
            self.points = extrapolate_points((7, 7), corners)
            #optional - show chessboard corners and smooths it on the window (the grid on frame)
            if SHOW_CHESS_MARKERS:
                fnl = cv2.drawChessboardCorners(img.copy(), (7, 7), corners, ret)
                self.show(WINDOW_MARKERS, fnl)
    #calibration process using aruco marker (ignore this part)
    def update_cal_quick(self, img):
        corners, ids = self.identify_aruco(img, False)
        if corners is None or ids is None:
            return

        grid = []
        for points, id_ in zip(corners, ids):
            if id_[0] not in CORNERS:
                continue

            centre = (np.sum(points, axis=1) / 4)[0]
            grid.append((centre, points[0]))

        if SHOW_QUICK_CAL:
            im = img.copy()
            aruco.drawDetectedMarkers(im, corners, ids)

        if len(grid) == 4:
            tl = corner_point(grid, (0, 0), get=lambda x: x[0])
            tr = corner_point(grid, (img.shape[1], 0), get=lambda x: x[0])
            bl = corner_point(grid, (0, img.shape[0]), get=lambda x: x[0])
            br = corner_point(grid, (img.shape[1], img.shape[0]), get=lambda x: x[0])

            tl_inner = corner_point(tl[1], br[0])
            tr_inner = corner_point(tr[1], bl[0])
            bl_inner = corner_point(bl[1], tr[0])
            br_inner = corner_point(br[1], tl[0])

            if SHOW_QUICK_CAL:
                for (start, end) in (
                    (tl_inner, tr_inner),
                    (tr_inner, br_inner),
                    (br_inner, bl_inner),
                    (bl_inner, tl_inner)
                ):
                    cv2.line(im, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 1)

            grid = interpolate_points((tl_inner, tr_inner, bl_inner, br_inner))
            self.points = grid

        if SHOW_QUICK_CAL:
            self.show(WINDOW_QUICK_CAL, im)
    #display focus region within input image.
    def show_focus_region(self, img):
        #check self points attributes
        if self.points:
            
            #why is pad 10?? Padding makes size larger
            pad = 10 
            focus = self.extract(
                img,
                np.float32(
                    [
                        self.points[0][0],
                        self.points[8][0],
                        self.points[0][8],
                        self.points[8][8],
                    ]
                ),
                #takes length of square times by 8. (full square in the chessboard parameter )
                self.square_w * 8,
                self.square_w * 8,
                pad,
            )
            if SHOW_EXTRAPOLATED:
                #visualize calibration point from chessboard using self.points method
                cb = draw_points(img, self.points)
                self.show(WINDOW_EXTRAP_BOARD, cb)
            #displays it with aruco chess piece (identify aruco markers in focus region)
            corners, ids = self.identify_aruco(focus, True)
            if SHOW_BOARD:
                #draws the detected aruco on focused region. Monitor positions in the board.
                aruco.drawDetectedMarkers(focus, corners, ids)

            for i in {**self.pieces}:
                #decrements and pops the element out (if counter == 0). Might be related to aruco markers?
                self.pieces[i][2] -= 1
                if self.pieces[i][2] == 0:
                    self.pieces.pop(i)

            if corners is not None and ids is not None:
                #binds aruco and ids together. Padding - simplify detection.
                #for loop gets the position of aruco markers and store in self.pieces
                for points, id_ in zip(corners, ids):
                    centre = (np.sum(points, axis=1) / 4)[0]
                    x = int((centre[0] - pad) // self.square_w)
                    y = int((centre[1] - pad) // self.square_w)
                    self.pieces[id_[0]] = [x, y, self.PIECES_WAIT] #wait is a counter - defined in the start of the code (15)

                    if SHOW_BOARD:
                        #draws circle around aruco marker. displays the x,y coordinates of aruco markets.
                        #operate within the focus region.
                        cv2.circle(focus, (int(centre[0]), int(centre[1])), 5, (0, 255, 0))
                        cv2.putText(
                            focus,
                            f"({x},{y})",
                            (int(centre[0]), int(centre[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )
            if SHOW_BOARD:
                self.show(WINDOW_EXTRA_BOARD, focus)

    def serialize(self):
        out = ""
        for y in range(8):
            #start loop for the x, y coordinates
            line = [] #stores empty position in an array.

            for x in range(8):
                for piece in self.pieces:
                    #take piece information
                    x1, y1, _ = self.pieces[piece]
                    if x1 == x and y1 == y:
                        #Match piece to position on a board. 
                        allocation = self.piece_allocations.get(piece)
                        if allocation is None:
                            #not found
                            line.append("??")
                        else:
                            #giving it color (Black or white) + type of pieces.
                            line.append(("B" if allocation[1] else "W") + "RBNKQP"[allocation[0]])
                        break #break loop, does per square per chess piece type
                else:
                    line.append("--") #if no piece found in square, return -- (no piece on that square)
            out += " ".join(line) + "\n" #adds all info in line array into serialized string with each piece information
        return out
    #GUI
    def flask_route(self):
        return self.serialize()
    #process input image, anotates chess pieces, updates the regions.
    def detect(self, img):
        if self.live_cal.get():
            if self.quick_cal.get():
                #if quick is true - execute update via quick call
                self.update_cal_quick(img)
            else:
                #if not - normal cal update
                self.update_cal(img)

        if self.points:
            #extract 8x8 (chess squares)
            regions = self.get_8x8_regions(img, self.points)
            #t
            for piece in self.pieces:
                (x, y, _) = self.pieces[piece]
                #annotates 8x8 region by coordinates. show coordinates on image via text.
                cv2.putText(
                    regions[x][y],
                    str(piece),
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 0, 0),
                    2,
                )
                #allocate individual pieces (like searialize function), show them via text.
                allocation = self.piece_allocations.get(piece)
                if allocation is not None:
                    cv2.putText(
                        regions[x][y],
                        "Black" if allocation[1] else "White",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 255),
                        2,
                    )
                    cv2.putText(
                        regions[x][y],
                        NAME[allocation[0]],
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
            #show focus region and process regions
            self.show_focus_region(img)
            self.process_regions(regions)

    def mainloop(self):
        #run app via thread (concurrent?) for Flask web-app
        threading.Thread(target=self.app.run, kwargs={"port":3000}, daemon=True).start()
        #Allow user to quit the program by turning running true to false.
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "running", False))

        #program runs
        while self.running:
            img = self.get_frame()
            self.detect(img)
            cv2.waitKey(1) #allow user input (1ms)
            self.render_tk() #render the GUI
        #if programs stops running - 'goodbye'
        print("Goodbye!")
        #destroy gui when program stops
        if self.tk is not None:
            self.tk.destroy()
        #release video
        self.vid.release()


if __name__ == "__main__":
    Main(int(input(">"))).mainloop()
