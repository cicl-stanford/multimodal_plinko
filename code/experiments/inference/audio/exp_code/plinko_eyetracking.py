from psychopy import visual, event, core
from psychopy.hardware import keyboard
from pygaze import libscreen, eyetracker, libtime
import pygaze
import pygame
import numpy as np
import datetime
import json
import os

trackertype = "dummy"
vid_length = 5000
testing = True
break_period = 3 if testing else 30
show_movies = True
col_demo = False

def collect_info():
    categories = ['first name', 'gender', 'age', 'race', 'ethnicity (Hispanic or non-Hispanic)']
    participant = {}
    for cat in categories:
        inp = input(f'Please type your {cat}:\n')
        if cat == 'ethnicity (Hispanic or non-Hispanic)':
            participant['ethnicity'] = inp
        elif cat == 'first name':
            participant['fname'] = inp
        else:
            participant[cat] = inp
    date = datetime.datetime.now()
    participant['date'] = f"{date.month}/{date.day}/{date.year}"
    return participant

def wait_loop(klist):
    event.clearEvents()
    while True:
        keys = event.getKeys(keyList=klist)
        if keys:
            return keys

def show_text(win, text, cont_inst=True, klist=['space']):
    if cont_inst:
        text += '\n\n(press space to continue)'
    inst = visual.TextStim(win, text=text, units='norm', height=0.1, wrapWidth=1.5)
    inst.draw()
    win.flip()
    wait_loop(klist)

def show_movie(win, path, proc):
    # Minimal, robust setup; audio off while debugging
    mov = visual.MovieStim(win, path, units='pix', size=(600, 500),
                           loop=False)
    mov.play()

    # All times are in SECONDS
    duration = getattr(mov, 'duration', 0.0) or 0.0
    have_dur = duration > 0
    pad_after_duration = 0.75     # extra slack after reported duration
    stagnation_grace   = 1.25     # no frame advance for this long => exit

    clock = core.MonotonicClock()
    start_t = clock.getTime()
    last_frameN = -1
    last_advance_t = start_t

    while True:
        mov.draw()
        win.flip()

        # 1) Exit when PsychoPy reports finished
        if getattr(mov, 'status', None) == visual.FINISHED:
            mov.draw()
            win.flip(clearBuffer=False)  # keep last frame visible
            break

        # Track progress to detect stagnation
        frameN = getattr(mov, 'frameN', None)
        if isinstance(frameN, int) and frameN != last_frameN:
            last_frameN = frameN
            last_advance_t = clock.getTime()

        # 2) Exit on stagnation *after* duration + pad
        t = clock.getTime() - start_t
        if have_dur and t > (duration + pad_after_duration):
            if (clock.getTime() - last_advance_t) > stagnation_grace:
                mov.draw()
                win.flip(clearBuffer=False)
                break

    mov.stop()
    # Do NOT immediately blank the screen here; draw the next thing instead.

    if proc:
        world_num = path.split("_")[1]  # assumes format "../videos/world_x_hole_y.mp4"
        last_path = f'../images/last_frames/last_frame_{world_num}.png'
        last_frame_img = visual.ImageStim(win, last_path, units='pix', size=(600, 500))
        proc_text = visual.TextStim(win, text='Press space to proceed',
                                    units='pix', pos=(0, -275), height=30)
        last_frame_img.draw(); proc_text.draw()
        win.flip()
        event.clearEvents()
        while not event.getKeys(keyList=['space']):
            core.wait(0.01)

def show_instruct(win):
    inst1 = ('In this experiment, we will ask you to make judgments about physical interactions. '
             'These interactions take place in a plinko box, a box with three holes on the top and obstacles inside. '
             'We drop a marble into one of the holes and it falls to the bottom of the plinko box bouncing off any obstacles in the way. '
             'Here are some examples to familiarize you with how the plinko box works:')
    show_text(win, inst1)
    video_loc = '../videos/'
    vids = [v for v in os.listdir(video_loc) if v.endswith('.mp4')]
    if show_movies:
        for vid in vids:
            show_movie(win, os.path.join(video_loc, vid), False)
            core.wait(3)
            show_movie(win, os.path.join(video_loc, vid), True)
    inst2 = ("Now that you have a sense for how the plinko box works, we can run some trials. In each trial, we will first cover up the box, and drop the ball into the box. You won't see the ball being dropped but hear the sounds that it makes as it collides with the obstacles, the walls, and as it lands in the sand at the bottom of the box.\n\nWe will then remove the cover of the box, and it's your job to figure out in which hole the ball was dropped. You will answer on the keypad using the number keys 1, 2, and 3.")
    show_text(win, inst2)


#This displays the cover image and plays a sound corresponding to the subsequent plinko box image
def disp_sound_cover(disp, image_name):

    #This first command sequence creates a substring that allows the sound to match its corresponding image 

    #print(image_name) #if you want to test if the image name matches sound name
    sound_name = ""
    #checks to see whether the pathway is through "trials" or "practice"
    if(image_name[10] == 't'):
        sound_name = image_name[16:-4]
    else:
        sound_name = image_name[19:-4]
        
    sound_path = '../sounds/' + sound_name + '.wav' 
    sound = pygame.mixer.Sound(sound_path)
    
    tr_sc = libscreen.Screen(disptype='psychopy')
    win = pygaze.expdisplay  # use the window owned by PyGaze

    cover_path = '../images/cover.png'
    cover = visual.ImageStim(win, cover_path, units='pix', size=(600, 500))
    num1 = visual.TextStim(win, text="1", units='pix', pos=(-148,270), height=40)
    num2 = visual.TextStim(win, text="2", units='pix', pos=(0, 270), height=40)
    num3 = visual.TextStim(win, text="3", units='pix', pos=(148,270), height=40)

    tr_sc.screen.append(cover)
    tr_sc.screen.append(num1)
    tr_sc.screen.append(num2)
    tr_sc.screen.append(num3)

    #place the screen in the display 
    disp.fill(screen=tr_sc)
    disp.show()

    #plays the sound and keeps the cover on the screen for 3 sec
    pygame.mixer.Sound.play(sound)

    #Ensures that the cover images stays on the screen while the sound plays
    while(pygame.mixer.get_busy()): 
        core.wait(0.5)

    #The cover image stays on the screen 1 sec after the sound stops
    core.wait(1)

def disp_im(disp, tracker, image_path):
    # Build a PyGaze Screen (psychopy backend)
    scr = libscreen.Screen(disptype='psychopy')

    # IMPORTANT: use the window owned by PyGaze
    win = pygaze.expdisplay

    # Make PsychoPy stimuli on that window
    num1 = visual.TextStim(win, text="1", units='pix', pos=(-148, 270), height=40)
    num2 = visual.TextStim(win, text="2", units='pix', pos=(   0, 270), height=40)
    num3 = visual.TextStim(win, text="3", units='pix', pos=( 148, 270), height=40)
    im   = visual.ImageStim(win, image=image_path, units='pix', size=(600, 500))

    # Append those PsychoPy stimuli to the PyGaze Screen
    scr.screen.append(num1)
    scr.screen.append(num2)
    scr.screen.append(num3)
    scr.screen.append(im)

    # Show the screen
    disp.fill(scr)
    disp.show()

    # Start eye tracking and sample until response
    tracker.start_recording()
    libtime.expstart()  # PyGaze clock in ms

    x_list, y_list, t_list, p_list = [], [], [], []
    last_ms = libtime.get_time()
    resp = None
    while resp is None:
        now = libtime.get_time()
        if now - last_ms >= 1.0:  # ~1kHz best effort
            x, y = tracker.sample()
            p = tracker.pupil_size()
            x_list.append(x if x is not None else float('nan'))
            y_list.append(y if y is not None else float('nan'))
            p_list.append(p if p is not None else float('nan'))
            t_list.append(now)
            last_ms = now

        keys = event.getKeys(['1', '2', '3', 'escape'])
        if keys:
            if 'escape' in keys:
                tracker.stop_recording()
                core.quit()
            resp = next((k for k in keys if k in ['1','2','3']), None)

        core.wait(0.0005)

    tracker.stop_recording()
    return resp, {'x': x_list, 'y': y_list, 't_ms': t_list, 'p': p_list}

def run_trial(disp, win, tracker, image):
    fix = visual.TextStim(win, text="+", units='norm', pos=(0, 0), height=0.2)
    fix.draw()
    win.flip()
    checked = False
    while not checked:
        checked = tracker.drift_correction()
    disp_sound_cover(disp, image)
    judgment, eye_info = disp_im(disp, tracker, image)
    return judgment, eye_info

def world_num(wstr):
    try:
        return int(wstr.split('_')[-1].split('.')[0])
    except Exception:
        raise Exception('Improper string format for file name ' + wstr)

def part_num():
    files = [f for f in os.listdir('Output') if f.endswith('.json')]
    nums = [int(f[3:-5]) for f in files if f.startswith('out')]
    return max(nums) + 1 if nums else 1

def run_experiment():
    if col_demo:
        part = part_num()
        demo = collect_info()
    else:
        part = part_num()
        demo = {}
    prac_path = '../images/practice/'
    practice = [os.path.join(prac_path, pic) for pic in os.listdir(prac_path)]
    trial_path = '../images/trial/'
    trials = [os.path.join(trial_path, pic) for pic in os.listdir(trial_path)]
    np.random.shuffle(trials)
    if testing:
        trials = trials[:9]
    win = visual.Window(size=(1024, 768), color='grey', units='pix', fullscr=False, allowGUI=True, waitBlanking=False, screen=0)
    show_instruct(win)
    disp = libscreen.Display(disptype='psychopy')
    tracker = eyetracker.EyeTracker(disp, trackertype=trackertype)
    tracker.calibrate()
    for stim in practice:
        run_trial(disp, win, tracker, stim)
    trial_data = []
    break_text = ("Time for a quick break. Feel free to blink your eyes and move about. "
                  "We'll come back and recalibrate when you are ready!")
    for i, tr in enumerate(trials):
        judgment, eye_info = run_trial(disp, win, tracker, tr)
        trial_data.append({'trial': world_num(tr), 'judgment': judgment, 'eye_data': eye_info})
        if i + 1 != len(trials) and (i + 1) % break_period == 0:
            show_text(pygaze.expdisplay, break_text)
            tracker.calibrate()
    show_text(pygaze.expdisplay, "That's all. Thanks for your participation!")
    tracker.close()
    win.close()
    exp_record = {'participant': part, 'demographics': demo, 'trials': trial_data}
    with open(f'Output/out{part}.json', 'w+') as f:
        json.dump(exp_record, f)
    return exp_record

if __name__ == "__main__":
    run_experiment()