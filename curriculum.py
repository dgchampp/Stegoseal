class CurriculumScheduler:
    def __init__(self, total_epochs, milestones=[0.2, 0.4, 0.6, 0.8]):
        self.total_epochs = total_epochs
        self.milestones = [int(total_epochs * m) for m in milestones]
        self.current_stage = 0
        self.distortion_levels = [0.3, 0.5, 0.7, 0.9, 1.0]
        
    def update(self, epoch):
        for i, milestone in enumerate(self.milestones):
            if epoch >= milestone and self.current_stage == i:
                self.current_stage = i + 1
                print(f"Entering curriculum stage {self.current_stage}: "
                      f"Severity {self.distortion_levels[self.current_stage]:.1f}")
    
    def get_distortion_level(self):
        return self.distortion_levels[self.current_stage]