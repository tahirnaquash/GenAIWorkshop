from constraint import Problem

# Inputs
classes = ["Class A", "Class B"]
rooms = ["Room 101", "Room 102", "Lab 1"]
timeslots = ["9AM", "10AM", "11AM"]
subjects = ["Math", "Physics", "Chemistry"]
staff = {
    "Math": "Mr. Rao",
    "Physics": "Ms. Nisha",
    "Chemistry": "Dr. Ahmed"
}

# Setup
problem = Problem()
for cls in classes:
    for subject in subjects:
        problem.addVariable((cls, subject), [(room, time) for room in rooms for time in timeslots])

# Constraints
def no_conflicts(*args):
    values = list(args)
    return len(values) == len(set(values))  # no two subjects in the same room at the same time

# Applying the constraint across all subjects in all classes
variables = [(cls, subject) for cls in classes for subject in subjects]
problem.addConstraint(no_conflicts, variables)

# Solve
solutions = problem.getSolutions()
for s in solutions[:1]:  # Show only the first timetable
    for k, v in s.items():
        cls, subject = k
        room, time = v
        teacher = staff[subject]
        print(f"{cls} - {subject} by {teacher} in {room} at {time}")