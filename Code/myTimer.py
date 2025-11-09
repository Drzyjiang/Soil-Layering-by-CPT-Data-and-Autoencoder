# This modulus aims at recording execution time

import time
#from datetime import timedelta

class executionTime:
    def __init__(self):
        self.startTime = 0
        self.endTime = 0
        self.elapsedTime = 0
    
    def startTimer(self):
        self.startTime = time.time()
    
    def endTimer(self):
        self.endTime = time.time()
    
    def calculateElapsedTime(self):
        self.elapsedTime = self.endTime - self.startTime
        hours, remainder = divmod(int(self.elapsedTime), 3600)
        minutes, seconds = divmod(remainder, 60)
        return hours, minutes, seconds

    def presentExecutionTime(self):
        hours, minutes, seconds = self.calculateElapsedTime()
        print(f"The total execution time is: {hours:02d}:{minutes:02d}:{seconds:02d}")

    
