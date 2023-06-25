from lib.cluster import Server

server = Server('batch/progress.txt')
server.train(secs = 5, fps = 1)