from timeit import default_timer as timer


class profiler():

    segments = []
    start = 0
    end = 0


    @staticmethod
    def set ():

        profiler.end = timer()
        profiler.segments.append(profiler.end - profiler.start)
        profiler.start = timer()

        return

    @staticmethod
    def init():
        profiler.start = timer()

        return

    @staticmethod
    def finish(profilerLog):
        profiler.end = timer()
        profiler.segments.append(profiler.end - profiler.start)
        str2log = ""
        for i in range(len(profiler.segments)):
            str2log += str(i) +"-"+str(i+1)+" = " + "{:.2f}".format(profiler.segments[i]) + " "
        profilerLog.debug(str2log)
        profiler.segments = []
        return

