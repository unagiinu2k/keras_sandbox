#http://d.hatena.ne.jp/naraba/20121201/p1
#http://web.mit.edu/music21/doc/usersGuide/usersGuide_01_installing.html

from music21 import *
environment.set('musicxmlPath' , r"C:\Program Files (x86)\Finale NotePad 2012\Finale NotePad.exe")
#configure.run()

s = corpus.parse('bach/bwv65.2.xml')
s.analyze('key')
s.show()

converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#").show()
s = converter.parse('/home/naraba/program/python/music21/etude10-05.xml')
s.show()
environment.keys()
environment.get('musicxmlPath')

