import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

base = 6
height = 4
area = 0.5 * base * height

def update(frame):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Triangle vertices
    p1 = (2, 2)
    p2 = (2 + base, 2)
    p3 = (2, 2 + height)

    # Draw triangle
    if frame >= 0:
        ax.add_patch(patches.Polygon([p1, p2, p3], closed=True, fill=True, color='skyblue', ec='black'))

    # Show base
    if frame >= 5:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=2)
        ax.text((p1[0] + p2[0]) / 2 - 0.5, p1[1] - 0.3, f'Base = {base}', fontsize=12)

    # Show height line
    if frame >= 10:
        ax.plot([p3[0], p1[0]], [p3[1], p1[1]], color='black', lw=2, ls='--')
        ax.text(p3[0] - 1.2, (p1[1] + p3[1]) / 2, f'Height = {height}', fontsize=12, rotation=90)

    # Step-by-step formula display
    if frame >= 15:
        ax.text(1, 6.5, 'Area = ½ × base × height', fontsize=14, color='darkblue')

    if frame >= 20:
        ax.text(1, 6.0, f'= ½ × {base} × {height}', fontsize=14, color='darkgreen')

    if frame >= 25:
        ax.text(1, 5.5, f'= {area}', fontsize=14, color='purple')

    if frame >= 30:
        ax.text(4, 7, f'Final Area: {area} units²', fontsize=14, color='crimson', weight='bold')

ani = animation.FuncAnimation(fig, update, frames=40, interval=400)
plt.show()